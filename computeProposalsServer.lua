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

local restserver = require("restserver")

local server = restserver:new():port(8080)

--------------------------------------------------------------------------------
-- parse arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('evaluate DeepCrop/SharpCrop')
cmd:text()
cmd:argument('-model', 'path to model to load')
cmd:text('Options:')
cmd:option('-gpu', 1, 'gpu device')
cmd:option('-si', -1.5, 'initial scale')
cmd:option('-sf', 1, 'final scale')
cmd:option('-ss', 0.5, 'scale step')
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
  np = 3,
  scales = scales,
  meanstd = meanstd,
  model = model,
  dm = config.dm,
}

function getProposals(img_path,cropClickX,cropClickY)
  -- load image
  local img = image.load(img_path)
  local h,w = img:size(2),img:size(3)

  local distanceInp = torch.FloatTensor(1,h,w)
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
  dists = (dists:div(dists:max())-0.5)*6
  distanceInp[1] = dists
  --Create combine 4 x wSz x wSz input
  local combinedInp = torch.cat(img,distanceInp,1)
  -- forward all scales
  infer:forward(combinedInp)
  -- get top propsals
  local masks,_ = infer:getTopProps(.2,h,w)
  -- save result
  local res = img:clone()
  maskApi.drawMasks(res, masks, 10)
  image.save(string.format('./res.jpg',config.model),res)
  return 'res.jpg'
end

--------------------------------------------------------------------------------
-- do it
print('| start')
server:add_resource("deepcrop", {
   {
      method = "GET",
      path = "/",
      input_schema = {
      },
      handler = function(req)
         print('GET!')
         return restserver.response():status(200):entity('Up!')
      end,
   },
   {
      method = "POST",
      path = "/",
      consumes = "application/json",
      produces = "application/json",
      input_schema = {
         img_64 = { type = "string" },
         x = { type = "number" },
         y = { type = "number" },
      },
      handler = function(req)
         print('POST!')
         local img_path = 'req_img.jpg'
         local decode_com = string.format('echo %s | base64 -d > %s',req.img_64,img_path)
         os.execute(decode_com)
         local res_path = getProposals(img_path,req.x,req.y)
         local encoded_com = io.popen(string.format('base64 %s',res_path))
         local encoded = encoded_com:read("*a")
         encoded_com:close()
         return restserver.response():status(200):entity(encoded)
      end,
   },
   
})

-- This loads the restserver.xavante plugin
server:enable("restserver.xavante"):start()

print('| done')
collectgarbage()
