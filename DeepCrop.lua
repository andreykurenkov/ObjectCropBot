--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

When initialized, it creates/load the common trunk, the maskBranch and the
scoreBranch.
DeepMask class members:
  - trunk: the common trunk (modified pre-trained resnet50)
  - maskBranch: the mask head architecture
  - scoreBranch: the score head architecture
------------------------------------------------------------------------------]]

require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
paths.dofile('SpatialSymmetricPadding.lua')
local utils = paths.dofile('modelUtils.lua')

local DeepMask,_ = torch.class('nn.DeepMask','nn.Container')

--------------------------------------------------------------------------------
-- function: constructor
function DeepMask:__init(config)
  -- create image features branch
  self.featuresBranch = self:createFeaturesBranch(config)

  -- create distance from crop pixel branch
  self.distanceBranch = self:createDistanceBranch(config)

  -- create mask head
  self.maskBranch = self:createMaskBranch(config)

  -- combine into a single model
  self.combinedModel = self:createCombinedModel(config)

  -- number of parameters
  local npt,nps,npm = 0,0,0
  local p1,p2,p3  = self.featuresBranch:parameters(),self.maskBranch:parameters(),self.distanceBranch:parameters()
  for k,v in pairs(p1) do npt = npt+v:nElement() end
  for k,v in pairs(p2) do npm = npm+v:nElement() end
  for k,v in pairs(p3) do nps = nps+v:nElement() end
  print(string.format('| number of paramaters features branch: %d', npt))
  print(string.format('| number of paramaters mask branch: %d', npm))
  print(string.format('| number of paramaters distanc bBranch branch: %d', nps))
  print(string.format('| number of paramaters total: %d', npt+nps+npm))
end

--------------------------------------------------------------------------------
-- function: create common trunk
function DeepMask:createFeaturesBranch(config)
  -- size of feature maps at end of trunk
  self.fSz = config.iSz/16

  -- load trunk
  local featuresBranch = torch.load('pretrained/resnet-50.t7')

  -- insert squeeze layer
  featuresBranch:insert(nn.Squeeze())

  -- remove BN
  utils.BNtoFixed(trunk, true)

  -- remove fully connected layers
  featuresBranch:featuresBranch();featuresBranch:remove();featuresBranch:remove();featuresBranch:remove()

  -- crop central pad
  featuresBranch:add(nn.SpatialZeroPadding(-1,-1,-1,-1))

  -- add common extra layers
  featuresBranch:add(cudnn.SpatialConvolution(1024,128,1,1,1,1))
  featuresBranch:add(cudnn.ReLU())
  featuresBranch:add(nn.View(config.batch,128*self.fSz*self.fSz))

  -- from scratch? reset the parameters
  if config.scratch then
    for k,m in pairs(trunk.modules) do if m.weight then m:reset() end end
  end

  -- symmetricPadding
  utils.updatePadding(featuresBranch, nn.SpatialSymmetricPadding)

  return featuresBranch:cuda()
end

--------------------------------------------------------------------------------
-- function: create distance branch
function DeepMask:createDistanceBranch(config)
  local distanceBranch = nn.Sequential()

  -- insert squeeze layer
  distanceBranch:insert(nn.Squeeze())
  distanceBranch:insert(nn.SpatialAdaptiveMaxPooling(self.fSz,self.fSz))
  distanceBranch:insert(nn.Unsqueeze(1))

  return distanceBranch:cuda()
end


--------------------------------------------------------------------------------
-- function: create mask branch
function DeepMask:createMaskBranch(config)
  local maskBranch = nn.Sequential()
  maskBranch:add(nn.Linear(129*self.fSz*self.fSz,512))

  -- maskBranch
  maskBranch:add(nn.Linear(512,config.oSz*config.oSz))
  maskBranch = nn.Sequential():add(maskBranch:cuda())

  -- upsampling layer
  if config.gSz > config.oSz then
    local upSample = nn.Sequential()
    upSample:add(nn.Copy('torch.CudaTensor','torch.FloatTensor'))
    upSample:add(nn.View(config.batch,config.oSz,config.oSz))
    upSample:add(nn.SpatialReSamplingEx{owidth=config.gSz,oheight=config.gSz,
    mode='bilinear'})
    upSample:add(nn.View(config.batch,config.gSz*config.gSz))
    upSample:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'))
    maskBranch:add(upSample)
  end

  return maskBranch
end

--------------------------------------------------------------------------------
-- function: create full model
function DeepMask:createCombinedModel(config)
  local combinedModel = nn.Sequential()
  local inputBranches = nn.Parallel(1,1)
  inputBranches:add(self.featuresBranch)
  inputBranches:add(self.distanceBranch)
  combinedModel:add(inputBranches)
  combinedModel:add(self.maskBranch)

  return combinedModel
end


--------------------------------------------------------------------------------
-- function: training
function DeepMask:training()
  self.featuresBranch:training(); self.distanceBranch:training(); self.maskBranch:training()
end

--------------------------------------------------------------------------------
-- function: evaluate
function DeepMask:evaluate()
  self.featuresBranch:evaluate(); self.distanceBranch:evaluate(); self.maskBranch:evaluate()
end

--------------------------------------------------------------------------------
-- function: to cuda
function DeepMask:cuda()
  self.featuresBranch:cuda(); self.distanceBranch:cuda(); self.maskBranch:cuda()
end

--------------------------------------------------------------------------------
-- function: to float
function DeepMask:float()
  self.featuresBranch:float(); self.distanceBranch:float(); self.maskBranch:float()
end

--------------------------------------------------------------------------------
-- function: inference (used for full scene inference)
function DeepMask:inference()
  self.featuresBranch:evaluate()
  self.maskBranch:evaluate()
  self.distanceBranch:evaluate()

  utils.linear2convTrunk(self.featuresBranch,self.fSz)
  utils.linear2convHead(self.maskBranch.modules[1])
  self.maskBranch = self.maskBranch.modules[1]

  self:cuda()
end

--------------------------------------------------------------------------------
-- function: clone
function DeepMask:clone(...)
  local f = torch.MemoryFile("rw"):binary()
  f:writeObject(self); f:seek(1)
  local clone = f:readObject(); f:close()

  if select('#',...) > 0 then
    clone.featuresBranch:share(self.featuresBranch,...)
    clone.maskBranch:share(self.maskBranch,...)
  end

  return clone
end

return nn.DeepMask
