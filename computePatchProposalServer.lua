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
cmd:text('evaluate deepmask/sharpmask')
cmd:text()
cmd:argument('-model', 'path to model to load')
cmd:text('Options:')
cmd:option('-img','data/testImage.jpg' ,'path/to/test/image')
cmd:option('-gpu', 1, 'gpu device')
cmd:option('-si', -2.5, 'initial scale')
cmd:option('-sf',  -0.75, 'final scale')
cmd:option('-ss', .25, 'scale step')
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
local scales = {}
for i = config.si,config.sf,config.ss do table.insert(scales,2^i) end

local infer = Infer{
  np = 2,
  scales = scales,
  meanstd = meanstd,
  model = model,
  dm = config.dm,
}

function getProposal(img_path)
  -- load image
  local img = image.load(img_path)
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
     for k=0,w*h-1 do if (M[k]==0 ) then O[k]=255 end end
  end
  local path = 'res-'..img_path
  image.save(path,res)
  return path
end

--------------------------------------------------------------------------------
-- do it
print('| start')
server:add_resource("cropbot", {
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
      },
      handler = function(req)
         local id = string.sub(""..math.random(),3,23)
         local img_path = string.format('req_img_%s.jpg',id)
         local decode_com = string.format('echo %s | base64 -d > %s',req.img_64,img_path)
         os.execute(decode_com)
         local res_path = getProposal(img_path)
         local encoded_com = io.popen('base64 '..res_path)
         local encoded = encoded_com:read("*a")
         encoded_com:close()
         os.execute("rm "..img_path)
         os.execute("rm "..res_path)
         return restserver.response():status(200):entity(encoded)
      end,
   },
   
})

-- This loads the restserver.xavante plugin
server:enable("restserver.xavante"):start()

collectgarbage()
