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
cmd:text('evaluate deepmask/sharpmask')
cmd:text()
cmd:argument('-model', 'path to model to load')
cmd:text('Options:')
cmd:option('-img','data/testImage.jpg' ,'path/to/test/image')
cmd:option('-gpu', 1, 'gpu device')
cmd:option('-dm', false, 'use DeepMask version of SharpMask')

local config = cmd:parse(arg)

--------------------------------------------------------------------------------
-- various initializations
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(config.gpu)

local coco = require 'coco'
local maskApi = coco.MaskApi

local meanstd = {mean = { 0.485, 0.456, 0.406 }, std = { 0.229, 0.224, 0.225 }}

--------------------------------------------------------------------------------
-- load moodel
paths.dofile('DeepMask.lua')
paths.dofile('SharpMask.lua')

print('| loading model file... ' .. config.model)
local m = torch.load(config.model..'/model.t7')
local model = m.model
model:inference(2)
model:cuda()

--------------------------------------------------------------------------------
-- create inference module
local scales = {}

if torch.type(model)=='nn.DeepMask' then
  paths.dofile('InferDeepMask.lua')
elseif torch.type(model)=='nn.SharpMask' then
  paths.dofile('InferSharpMask.lua')
end

local infer = Infer{
  np = 2,
  scales = {0.75,1,1.25},
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

-- forward all scales
infer:forward(img)

-- get top propsals
local masks,_ = infer:getTopProps(.2,h,w)

-- save result
local res = img:clone()

local M = masks[1]:contiguous():data()
for j=1,3 do
 local O= res[j]:data()
 for k=0,w*h-1 do if M[k]==0 then O[k]=0 end end
end
image.save(string.format('./res.jpg',config.model),res)

print('| done')
collectgarbage()
