--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Training and testing loop for DeepCrop
------------------------------------------------------------------------------]]

local optim = require 'optim'
local image = require 'image'
paths.dofile('trainMeters.lua')

local Trainer = torch.class('Trainer')

--------------------------------------------------------------------------------
-- function: init
function Trainer:__init(model, criterion, config)
  -- training params
  self.config = config
  self.model = model
  self.combinedNet = model.combinedModel
  self.criterion = criterion
  self.lr = config.lr
  self.optimState ={}
  for k,v in pairs({'features','mask'}) do
    self.optimState[v] = {
      learningRate = config.lr,
      learningRateDecay = 0,
      momentum = config.momentum,
      dampening = 0,
      weightDecay = config.wd,
    }
  end

  -- params and gradparams
  self.pt,self.gt = model.featuresBranch:getParameters()
  self.pm,self.gm = model.maskBranch:getParameters()

  -- allocate cuda tensors
  self.inputs, self.labels = torch.CudaTensor(), torch.CudaTensor()

  -- meters
  self.lossmeter  = LossMeter()
  self.trainIouMeter  = IouMeter(0,config.maxload*config.batch)
  self.testIouMeter  = IouMeter(0,config.testmaxload*config.batch)

  -- log
  self.modelsv = {model=model:clone('weight', 'bias'),config=config}
  self.rundir = config.rundir
  self.log = torch.DiskFile(self.rundir .. '/log', 'rw'); self.log:seekEnd()
end

--------------------------------------------------------------------------------
-- function: train
function Trainer:train(epoch, dataloader)
  self.model:training()
  self:updateScheduler(epoch)
  self.trainIouMeter:reset()
  self.lossmeter:reset()

  local timer = torch.Timer()

  local fevalfeatures = function() return self.model.featuresBranch.output, self.gt end
  local fevalmask  = function() return self.criterion.output,   self.gm end
  print(string.format('[train] Starting training epoch of %d batches',dataloader:size()))
  for n, sample in dataloader:run() do
    if n%10==0 then
        print(string.format('[train] batch %d | s/batch %04.2f | loss: %07.5f ',n,timer:time().real/n,self.lossmeter:value()))
    end
    -- copy samples to the GPU
    self:copySamples(sample)

    local status, outputs = pcall(
        function() return self.combinedNet:forward(self.inputs) end)
    if not status then
      print('[train] Error during forward pass!!! ')
      print(outputs)
      --print(debug.traceback())
    else
      self.trainIouMeter:add(outputs:view(self.labels:size()),self.labels)
      local lossbatch = self.criterion:forward(outputs, self.labels)
      
      self.combinedNet:zeroGradParameters()
      local gradOutputs = self.criterion:backward(outputs, self.labels):mul(self.inputs:size(1))
      self.combinedNet:backward(self.inputs, gradOutputs)

      -- optimize
      optim.sgd(fevalfeatures, self.pt, self.optimState.features)
      optim.sgd(fevalmask, self.pm, self.optimState.mask)

      -- update loss
      self.lossmeter:add(lossbatch)

      if n<4 or n%500==0 then
        image.save(string.format('%%s/samples/train/train_%d_%d_in_img.jpg',config.rundir,epoch,n),self.inputs[1]:select(4,1))
        image.save(string.format('%%s/samples/train/train_%d_%d_in_dist.jpg',config.rundir,epoch,n),self.inputs[1][1]:select(3,2):add(1):div(2))
        image.save(string.format('%s/samples/train/train_%d_%d_in_dist2.jpg',config.rundir,epoch,n),self.inputs[1][2]:select(3,2))
        image.save(string.format('%s/samples/train/train_%d_%d_in_dist3.jpg',config.rundir,epoch,n),self.inputs[1][3]:select(3,2))
        labelSize = self.labels[1]:size()
        image.save(string.format('%s/samples/train/train_%d_%d_labels.jpg',config.rundir,epoch,n),self.labels[1]:resize(1,labelSize[1],labelSize[2]))
        image.save(string.format('%s/samples/train/train_%d_%d_out.jpg',config.rundir,epoch,n),outputs[1]:resize(1,labelSize[1],labelSize[2]):gt(0))
        print(string.format('[train] Saving samples - output: batch %d, output mean %04.3f, std %04.3f, max %04.3f, min %04.3f',n, outputs:mean(), outputs: std(), outputs:max(), outputs:min()))
      end
    end
  end

  -- write log
  local logepoch =
    string.format('[train] | epoch %05d | s/batch %04.2f | loss: %07.5f ',
      epoch, timer:time().real/dataloader:size(),self.lossmeter:value())
  print(logepoch)
  self.log:writeString(string.format('%s\n',logepoch))
  self.log:synchronize()

  --save model
  torch.save(string.format('%s/model.t7', self.rundir),self.modelsv)
  if epoch%50 == 0 then
    torch.save(string.format('%s/model_%d.t7', self.rundir, epoch),
      self.modelsv)
  end

  collectgarbage()
end

--------------------------------------------------------------------------------
-- function: test
local maxacc = 0
function Trainer:test(epoch, dataloader)
  self.model:evaluate()
  self.testIouMeter:reset()
  
  local timer = torch.Timer()
  print(string.format('[test] Starting testing epoch of %d batches',dataloader:size()))
  for n, sample in dataloader:run() do
    if n%10==0 then
        print(string.format('[test] batch %d | s/batch %04.2f | mean: %06.2f ',n,timer:time().real/n,self.testIouMeter:value('mean')))
    end
    -- copy input and target to the GPU
    self:copySamples(sample)
    local status, outputs = pcall(
        function() return self.combinedNet:forward(self.inputs) end)
    if not status then
      print('[test] Error during forward pass!!! ')
      print(outputs)
      --print(debug.traceback())
    else
      self.testIouMeter:add(outputs:view(self.labels:size()),self.labels)
      cutorch.synchronize()
   
      if n<4 then
        image.save(string.format('%s/samples/test/test_%d_%d_in_img.jpg',config.rundir,epoch,n),self.inputs[1]:select(4,1))
        image.save(string.format('%s/samples/test/test_%d_%d_in_dist.jpg',config.rundir,epoch,n),self.inputs[1][1]:select(3,2):add(1):div(2))
        image.save(string.format('%s/samples/test/test_%d_%d_in_dist2.jpg',config.rundir,epoch,n),self.inputs[1][2]:select(3,2))
        image.save(string.format('%s/samples/test/test_%d_%d_in_dist3.jpg',config.rundir,epoch,n),self.inputs[1][3]:select(3,2))
        labelSize = self.labels[1]:size()
        image.save(string.format('%s/samples/test/test_%d_%d_labels.jpg',config.rundir,epoch,n),self.labels[1]:resize(1,labelSize[1],labelSize[2]))
        image.save(string.format('%s/samples/test/test_%d_%d_out.jpg',config.rundir,epoch,n),outputs[1]:resize(labelSize[1],labelSize[2]):gt(0))
        print(string.format('[test] Saving samples - output: batch %d, output mean %04.3f, std %04.3f, max %04.3f, min %04.3f',n, outputs:mean(), outputs: std(), outputs:max(), outputs:min()))
      end
    end
  end
  self.model:training()

  -- check if bestmodel so far
  local z,bestmodel = self.testIouMeter:value('0.7')
  if z > maxacc then
    torch.save(string.format('%s/bestmodel.t7', self.rundir),self.modelsv)
    maxacc = z
    bestmodel = true
  end

  -- write log
  local logepoch =
    string.format('[test]  | epoch %05d '..
      '| IoU: mean %06.2f median %06.2f suc@.5 %06.2f | bestmodel %s',
      epoch,
      self.testIouMeter:value('mean'),self.testIouMeter:value('median'),
      self.testIouMeter:value('0.5'), self.testIouMeter:value('0.7'),
      bestmodel and '*' or 'x')
  print(logepoch)
  self.log:writeString(string.format('%s\n',logepoch))
  self.log:synchronize()

  collectgarbage()
end

--------------------------------------------------------------------------------
-- function: copy inputs/labels to CUDA tensor
function Trainer:copySamples(sample)
  self.inputs:resize(sample.inputs:size()):copy(sample.inputs)
  self.labels:resize(sample.labels:size()):copy(sample.labels)
end

--------------------------------------------------------------------------------
-- function: update training schedule according to epoch
function Trainer:updateScheduler(epoch)
  if self.lr == 0 then
    local regimes = {
      {   1,  50, 1e-3, 5e-4},
      {  51, 120, 5e-4, 5e-4},
      { 121, 1e8, 1e-4, 5e-4}
    }

    for _, row in ipairs(regimes) do
      if epoch >= row[1] and epoch <= row[2] then
        for k,v in pairs(self.optimState) do
          v.learningRate=row[3]; v.weightDecay=row[4]
        end
      end
    end
  end
end

return Trainer
