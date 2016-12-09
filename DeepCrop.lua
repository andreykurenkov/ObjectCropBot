--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

When initialized, it creates/load the common trunk, the maskBranch and the
scoreBranch.
DeepCrop class members:
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

local DeepCrop,_ = torch.class('nn.DeepCrop','nn.Container')

--------------------------------------------------------------------------------
-- function: constructor
function DeepCrop:__init(config)
  -- create image features branch
  self.featuresBranch = self:createFeaturesBranch(config)

  -- create distance from crop pixel branch
  self.distanceBranch = self:createDistanceBranch(config)

  -- create mask head
  self.maskBranch = self:createMaskBranch(config)
  
  -- create mask head
  self.scoreBranch = self:createScoreBranch(config)

  -- create paralelle branch 
  self.trunk = self:createCombinedFeatures(config)
  
  -- combine into final models
  self.maskModel = self:createMaskModel(config)
  self.scoreModel = self:createScoreModel(config)

  -- number of parameters
  local npt,nps,npm = 0,0,0
  local p1,p2,p3  = self.featuresBranch:parameters(),self.maskBranch:parameters(),self.distanceBranch:parameters()
  for k,v in pairs(p1) do npt = npt+v:nElement() end
  for k,v in pairs(p2) do npm = npm+v:nElement() end
  for k,v in pairs(p3) do nps = nps+v:nElement() end
  print(string.format('| number of paramaters features branch: %d', npt))
  print(string.format('| number of paramaters mask branch: %d', npm))
  print(string.format('| number of paramaters distance: %d', nps))
  print(string.format('| number of paramaters total: %d', npt+nps+npm))
end

--------------------------------------------------------------------------------
-- function: create common trunk
function DeepCrop:createFeaturesBranch(config)
  -- size of feature maps at end of trunk
  self.fSz = config.iSz/16

  local featuresBranch = torch.load('pretrained/resnet-18.t7')
  -- load trunk
  --local featuresBranch = torch.load('pretrained/resnet-50.t7')
  -- remove BN
  utils.BNtoFixed(featuresBranch, true)

  -- remove fully connected layers
  featuresBranch:remove()
  featuresBranch:remove()
  featuresBranch:remove()
  featuresBranch:remove()

  -- crop central pad
  featuresBranch:add(nn.SpatialZeroPadding(-1,-1,-1,-1))

  -- add common extra layers
  featuresBranch:add(cudnn.SpatialConvolution(256,128,1,1,1,1))
  --featuresBranch:add(cudnn.SpatialConvolution(1024,128,1,1,1,1))
  featuresBranch:add(cudnn.ReLU())
  featuresBranch:add(nn.View(config.batch,128*self.fSz*self.fSz))

  -- from scratch? reset the parameters
  if config.scratch then
    for k,m in pairs(featuresBranch.modules) do if m.weight then m:reset() end end
  end

  -- symmetricPadding
  utils.updatePadding(featuresBranch, nn.SpatialSymmetricPadding)

  --Needed for backward step from parallel container
  featuresBranch:add(nn.Copy(nil,nil,true))
  return featuresBranch:cuda()
end

--------------------------------------------------------------------------------
-- function: create distance branch
function DeepCrop:createDistanceBranch(config)
  local distanceBranch = nn.Sequential()
  local k = config.iSz + 32 - self.fSz + 1 
  distanceBranch:insert(nn.SpatialSubSampling(3,k,k))
  --Needed for backward step from parallel container
  distanceBranch:add(nn.Reshape(3*self.fSz*self.fSz,true))

  return distanceBranch:cuda()
end

--------------------------------------------------------------------------------
-- function: create mask branch
function DeepCrop:createMaskBranch(config)
  local maskBranch = nn.Sequential()

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
-- function: create score branch
function DeepCrop:createScoreBranch(config)
  local scoreBranch = nn.Sequential()
  scoreBranch:add(nn.Dropout(.5))
  scoreBranch:add(nn.Linear(512,1024))
  scoreBranch:add(nn.Threshold(0, 1e-6))

  scoreBranch:add(nn.Dropout(.5))
  scoreBranch:add(nn.Linear(1024,1))

  return scoreBranch:cuda()
end

--------------------------------------------------------------------------------
-- function: create full mask model
function DeepCrop:createCombinedFeatures(config)
  local combinedModel = nn.Sequential()
  
  local inputBranches = nn.Parallel(5,2)
  inputBranches:add(self.featuresBranch)
  inputBranches:add(self.distanceBranch)
 
  combinedModel:add(inputBranches)
  -- 128 from feature branch and 3 from distance branch
  combinedModel:add(nn.Linear((128+3)*self.fSz*self.fSz,512))
  
  return combinedModel:cuda()
end

--------------------------------------------------------------------------------
-- function: create full mask model
function DeepCrop:createMaskModel(config)
  local combinedModel = nn.Sequential()
  combinedModel:add(self.trunk)
  combinedModel:add(self.maskBranch)
  
  return combinedModel
end

--------------------------------------------------------------------------------
-- function: create full score model
function DeepCrop:createScoreModel(config)
  local combinedModel = nn.Sequential()
  combinedModel:add(self.trunk)
  combinedModel:add(self.scoreBranch)
  
  return combinedModel
end
--------------------------------------------------------------------------------
-- function: training
function DeepCrop:training()
  self.trunk:training(); self.distanceBranch:training(); self.maskBranch:training(); self.scoreBranch:training()
end

--------------------------------------------------------------------------------
-- function: evaluate
function DeepCrop:evaluate()
  self.trunk:evaluate(); self.distanceBranch:evaluate(); self.maskBranch:evaluate(); self.scoreBranch:training()
end

--------------------------------------------------------------------------------
-- function: to cuda
function DeepCrop:cuda()
  self.trunk:cuda(); self.distanceBranch:cuda(); self.maskBranch:cuda(); self.scoreBranch:cuda()
end

--------------------------------------------------------------------------------
-- function: to float
function DeepCrop:float()
  self.trunk:float(); self.distanceBranch:float(); self.maskBranch:float(); self.scoreBranch:float()
end

--------------------------------------------------------------------------------
-- function: inference (used for full scene inference)
function DeepCrop:inference()
  self:evaluate()

  utils.linear2convTrunk(self.featuresBranch,self.fSz)
  utils.linear2convHead(self.maskBranch.modules[1])
  self.maskBranch = self.maskBranch.modules[1]

  self:cuda()
end

--------------------------------------------------------------------------------
-- function: clone
function DeepCrop:clone(...)
  local f = torch.MemoryFile("rw"):binary()
  f:writeObject(self); f:seek(1)
  local clone = f:readObject(); f:close()

  if select('#',...) > 0 then
    clone.featuresBranch:share(self.featuresBranch,...)
    clone.maskBranch:share(self.maskBranch,...)
  end

  return clone
end

return nn.DeepCrop
