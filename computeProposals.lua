--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Run full scene inference in sample image
------------------------------------------------------------------------------]]

require 'torch'
require 'cutorch'
require 'image'

--------------------------------------------------------------------------------
-- parse arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('evaluate DeepCrop/SharpCrop')
cmd:text()
cmd:argument('-model', 'path to model to load')
cmd:argument('-img', 'path/to/test/image')
cmd:argument('-clickX' ,'4')
cmd:argument('-clickY' ,'54')
cmd:text('Options:')
cmd:option('-gpu', 1, 'gpu device')
cmd:option('-np', 5,'number of proposals to save in test')
cmd:option('-si', -2.5, 'initial scale')
cmd:option('-sf', .5, 'final scale')
cmd:option('-ss', .5, 'scale step')
cmd:option('-dm', false, 'use DeepCrop version of SharpCrop')

local config = cmd:parse(arg)

--------------------------------------------------------------------------------
-- various initializations
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(config.gpu)

local coco = require 'coco'
local maskApi = coco.MaskApi

local meanstd = {mean = { 0.485, 0.456, 0.406 }, std = { 0.229, 0.224, 0.225 }}

--------------------------------------------------------------------------------
-- load model
paths.dofile('DeepCrop.lua')
paths.dofile('SharpCrop.lua')

print('| loading model file... ' .. config.model)
local m = torch.load(config.model..'/model.t7')
local model = m.model
model:inference(config.np)
model:cuda()

--------------------------------------------------------------------------------
-- create inference module
local scales = {}
for i = config.si,config.sf,config.ss do table.insert(scales,2^i) end

if torch.type(model)=='nn.DeepCrop' then
  paths.dofile('InferDeepCrop.lua')
elseif torch.type(model)=='nn.SharpCrop' then
  paths.dofile('InferSharpCrop.lua')
end

local infer = Infer{
  np = config.np,
  scales = scales,
  meanstd = meanstd,
  model = model,
  dm = config.dm,
}

--------------------------------------------------------------------------------
-- do it
print('| start')

-- load image
local img = image.load(config.img)
local h,w = img:size(2),img:size(3)
local cropClickX = config.clickX
local cropClickY = config.clickY

local distanceInp = torch.FloatTensor(3,h,w)
local pixel = img[{{1,3},cropClickY,cropClickX}]
img[{{1,3},cropClickX,cropClickY}] = torch.Tensor({0.111,0.222,0.333})
-- Calculate location difference from click pixel, via 2 norm
local pixels = torch.Tensor(torch.linspace(1,h,w))
local pixelsX = torch.Tensor(torch.linspace(1,w,w)):reshape(w,1):repeatTensor(1,h)
local pixelsY = torch.Tensor(torch.linspace(1,h,h)):reshape(1,h):repeatTensor(w,1)
local coords = pixelsX:cat(pixelsY,3):transpose(1,3)
local clickXs = torch.Tensor({cropClickX}):repeatTensor(w,h)
local clickYs = torch.Tensor({cropClickY}):repeatTensor(w,h)
local clickCoords = clickXs:cat(clickYs,3):transpose(1,3)
local dists = (coords-clickCoords):norm(2,1)
dists = (dists:div(dists:max())-0.5)*2
distanceInp[1] = dists

-- Calculate rgb difference from click pixel, via 2 nom
pixels = pixel:reshape(1,1,3):repeatTensor(w,h,1):transpose(1,3)
distanceInp[2] = (img-pixels):norm(2,1)

-- Calculate lum difference from click pixel, via 2 nom
lumTensor = torch.Tensor({0.299,0.587,0.114}):reshape(3,1,1)
imgLum = img:conv3(lumTensor)
pixelLum = pixels:conv3(lumTensor)
distanceInp[3] = (imgLum-pixelLum):norm(2,1)

--Create combine 3 x wSz x wSz input x 2
local combinedInp = torch.cat(img,distanceInp,4)


-- forward all scales
infer:forward(combinedInp)

-- get top propsals
local masks,_ = infer:getTopProps(.2,h,w)

-- save result
local res = img:clone()
maskApi.drawMasks(res, masks, 10)
image.save(string.format('./res.jpg',config.model),res)

print('| done')
collectgarbage()
