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
  self:createMaskBranch(config)

  -- combine into a single model
  self:createCombinedModel(config)

  -- number of parameters
  local npt,nps,npm = 0,0,0
  local p1,p2,p3  = self.trunk:parameters(),
    self.maskBranch:parameters(),self.scoreBranch:parameters()
  for k,v in pairs(p1) do npt = npt+v:nElement() end
  for k,v in pairs(p2) do npm = npm+v:nElement() end
  for k,v in pairs(p3) do nps = nps+v:nElement() end
  print(string.format('| number of paramaters trunk: %d', npt))
  print(string.format('| number of paramaters mask branch: %d', npm))
  print(string.format('| number of paramaters score branch: %d', nps))
  print(string.format('| number of paramaters total: %d', npt+nps+npm))
end

--------------------------------------------------------------------------------
-- function: create common trunk
function DeepMask:createFeaturesBranch(config)
  -- size of feature maps at end of trunk
  self.fSz = config.iSz/16

  -- load trunk
  local trunk = torch.load('pretrained/resnet-50.t7')

  -- insert squeeze layer
  trunk:insert(nn.Squeeze())

  -- remove BN
  utils.BNtoFixed(trunk, true)

  -- remove fully connected layers
  trunk:remove();trunk:remove();trunk:remove();trunk:remove()

  -- crop central pad
  trunk:add(nn.SpatialZeroPadding(-1,-1,-1,-1))

  -- add common extra layers
  trunk:add(cudnn.SpatialConvolution(1024,128,1,1,1,1))
  trunk:add(cudnn.ReLU())
  trunk:add(nn.View(config.batch,128*self.fSz*self.fSz))

  -- from scratch? reset the parameters
  if config.scratch then
    for k,m in pairs(trunk.modules) do if m.weight then m:reset() end end
  end

  -- symmetricPadding
  utils.updatePadding(trunk, nn.SpatialSymmetricPadding)

  return trunk:cuda()
end

--------------------------------------------------------------------------------
-- function: create distance branch
function DeepMask:createDistanceBranch(config)
  local distanceBranch = nn.Sequential()

  -- insert squeeze layer
  distanceBranch:insert(nn.Squeeze())
  distanceBranch:insert(nn.SpatialAdaptiveMaxPooling(self.fSz,self.fSz))
  distanceBranch:insert(nn.Unsqueeze(1))

  return nn.Sequential():add(distanceBranch:cuda())
end


--------------------------------------------------------------------------------
-- function: create mask branch
function DeepMask:createMaskBranch(config)
  local maskBranch = nn.Sequential()
  maskBranch:add(nn.Linear(129*self.fSz*self.fSz,512))

  -- maskBranch
  maskBranch:add(nn.Linear(512,config.oSz*config.oSz))
  self.maskBranch = nn.Sequential():add(maskBranch:cuda())

  -- upsampling layer
  if config.gSz > config.oSz then
    local upSample = nn.Sequential()
    upSample:add(nn.Copy('torch.CudaTensor','torch.FloatTensor'))
    upSample:add(nn.View(config.batch,config.oSz,config.oSz))
    upSample:add(nn.SpatialReSamplingEx{owidth=config.gSz,oheight=config.gSz,
    mode='bilinear'})
    upSample:add(nn.View(config.batch,config.gSz*config.gSz))
    upSample:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'))
    self.maskBranch:add(upSample)
  end

  return self.maskBranch
end

--------------------------------------------------------------------------------
-- function: create full model
function DeepMask:createCombinedModel(config)
  local combinedModel = nn.Sequential()
  local inputBranches = nn.Parallel(1,1)
  inputBranches:add(self.)

  -- maskBranch
  maskBranch:add(nn.Linear(512,config.oSz*config.oSz))
  self.maskBranch = nn.Sequential():add(maskBranch:cuda())

  -- upsampling layer
  if config.gSz > config.oSz then
    local upSample = nn.Sequential()
    upSample:add(nn.Copy('torch.CudaTensor','torch.FloatTensor'))
    upSample:add(nn.View(config.batch,config.oSz,config.oSz))
    upSample:add(nn.SpatialReSamplingEx{owidth=config.gSz,oheight=config.gSz,
    mode='bilinear'})
    upSample:add(nn.View(config.batch,config.gSz*config.gSz))
    upSample:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'))
    self.maskBranch:add(upSample)
  end

  return self.maskBranch
end


--------------------------------------------------------------------------------
-- function: training
function DeepMask:training()
  self.trunk:training(); self.maskBranch:training(); self.scoreBranch:training()
end

--------------------------------------------------------------------------------
-- function: evaluate
function DeepMask:evaluate()
  self.trunk:evaluate(); self.maskBranch:evaluate(); self.scoreBranch:evaluate()
end

--------------------------------------------------------------------------------
-- function: to cuda
function DeepMask:cuda()
  self.trunk:cuda(); self.scoreBranch:cuda(); self.maskBranch:cuda()
end

--------------------------------------------------------------------------------
-- function: to float
function DeepMask:float()
  self.trunk:float(); self.scoreBranch:float(); self.maskBranch:float()
end

--------------------------------------------------------------------------------
-- function: inference (used for full scene inference)
function DeepMask:inference()
  self.trunk:evaluate()
  self.maskBranch:evaluate()
  self.scoreBranch:evaluate()

  utils.linear2convTrunk(self.trunk,self.fSz)
  utils.linear2convHead(self.scoreBranch)
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
    clone.trunk:share(self.trunk,...)
    clone.maskBranch:share(self.maskBranch,...)
    clone.scoreBranch:share(self.scoreBranch,...)
  end

  return clone
end

return nn.DeepMask
