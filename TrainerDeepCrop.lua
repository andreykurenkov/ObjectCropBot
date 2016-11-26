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
  self.maskNet = model.maskModel
  self.scoreNet = model.scoreModel
  self.criterion = criterion
  self.lr = config.lr
  self.optimState ={}
  for k,v in pairs({'features','mask','score'}) do
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
  self.ps,self.gs = model.scoreBranch:getParameters()

  -- allocate cuda tensors
  self.inputs, self.labels = torch.CudaTensor(), torch.CudaTensor()

  -- meters
  self.lossmeter  = LossMeter()
  self.maskmeter  = IouMeter(0.5,config.testmaxload*config.batch)
  self.scoremeter = BinaryMeter()

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
  self.lossmeter:reset()

  local timer = torch.Timer()

  local fevalfeatures = function() return self.model.featuresBranch.output, self.gt end
  local fevalmask  = function() return self.criterion.output,   self.gm end
  local fevalscore = function() return self.criterion.output, self.gs end
  
  print(string.format('[train] Starting training epoch of %d batches',dataloader:size()))
  for n, sample in dataloader:run() do
    if self.config.verbose and n%10==0 then
        print(string.format('[train] batch %d | s/batch %04.2f | loss: %07.5f ',n,timer:time().real/n,self.lossmeter:value()))
    end
    -- copy samples to the GPU
    self:copySamples(sample)
    -- forward/backward
    local model, params, feval, optimState
    if sample.head == 1 then
      model, params = self.maskNet, self.pm
      feval,optimState = fevalmask, self.optimState.mask
    else
      model, params = self.scoreNet, self.ps
      feval,optimState = fevalscore, self.optimState.score
    end

    local status, outputs = pcall(
        function() return model:forward(self.inputs) end)
    if not status then
      print('[train] Error during forward pass!!! ')
      print(outputs)
      --print(debug.traceback())
    else
      local lossbatch = self.criterion:forward(outputs, self.labels)
      if outputs:mean()~=outputs:mean() then
        print('her!')
        print(sample.head)
      end   
      if lossbatch~=lossbatch  then
        print('heasdfar!')
        print(sample.head)
      end   
      model:zeroGradParameters()
      local gradOutputs = self.criterion:backward(outputs, self.labels)
      if sample.head == 1 then gradOutputs:mul(self.inputs:size(1)) end
      model:backward(self.inputs, gradOutputs)

      -- optimize
      optim.sgd(fevalfeatures, self.pt, self.optimState.trunk)
      optim.sgd(feval, params, optimState)

      -- update loss
      self.lossmeter:add(lossbatch)
      if self.config.verbose and (n<4 or n%500==0) then
        image.save(string.format('%s/samples/train/train_%d_%d_in_img.jpg',self.rundir,epoch,n),self.inputs[1]:select(4,1))
        image.save(string.format('%s/samples/train/train_%d_%d_in_dist.jpg',self.rundir,epoch,n),self.inputs[1][1]:select(3,2):add(1):div(2))
        image.save(string.format('%s/samples/train/train_%d_%d_in_dist2.jpg',self.rundir,epoch,n),self.inputs[1][2]:select(3,2))
        image.save(string.format('%s/samples/train/train_%d_%d_in_dist3.jpg',self.rundir,epoch,n),self.inputs[1][3]:select(3,2))
        if sample.head==1 then
          labelSize = self.labels[1]:size()
          image.save(string.format('%s/samples/train/train_%d_%d_labels.jpg',self.rundir,epoch,n),self.labels[1]:resize(1,labelSize[1],labelSize[2]))
           image.save(string.format('%s/samples/train/train_%d_%d_out.jpg',self.rundir,epoch,n),outputs[1]:resize(1,labelSize[1],labelSize[2]))
        end
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
  self.maskmeter:reset()
  self.scoremeter:reset()
  
  local timer = torch.Timer()
  print(string.format('[test] Starting testing epoch of %d batches',dataloader:size()))
  for n, sample in dataloader:run() do
    if self.config.verbose and n%10==0 then
        print(string.format('[test] batch %d | s/batch %04.2f | mean: %06.2f ',n,timer:time().real/n,self.maskmeter:value('mean')))
    end
    -- copy input and target to the GPU
    self:copySamples(sample)
    -- forward/backward
    local model
    if sample.head == 1 then
      model, meter = self.maskNet, self.maskmeter
    else
      model, meter = self.scoreNet, self.scoremeter
    end

    local status, outputs = pcall(
        function() return model:forward(self.inputs) end)
    if not status then
      print('[test] Error during forward pass!!! ')
      print(outputs)
      --print(debug.traceback())
    else
      meter:add(outputs:view(self.labels:size()),self.labels)
      cutorch.synchronize()
   
      if self.config.verbose and n<4 then
        image.save(string.format('%s/samples/test/test_%d_%d_in_img.jpg',self.rundir,epoch,n),self.inputs[1]:select(4,1))
        image.save(string.format('%s/samples/test/test_%d_%d_in_dist.jpg',self.rundir,epoch,n),self.inputs[1][1]:select(3,2):add(1):div(2))
        image.save(string.format('%s/samples/test/test_%d_%d_in_dist2.jpg',self.rundir,epoch,n),self.inputs[1][2]:select(3,2))
        image.save(string.format('%s/samples/test/test_%d_%d_in_dist3.jpg',self.rundir,epoch,n),self.inputs[1][3]:select(3,2))
        if sample.head==1 then
          labelSize = self.labels[1]:size()
          image.save(string.format('%s/samples/test/test_%d_%d_labels.jpg',self.rundir,epoch,n),self.labels[1]:resize(1,labelSize[1],labelSize[2]))
          image.save(string.format('%s/samples/test/test_%d_%d_out.jpg',self.rundir,epoch,n),outputs[1]:resize(labelSize[1],labelSize[2]))
        end
        print(string.format('[test] Saving samples - output: batch %d, output mean %04.3f, std %04.3f, max %04.3f, min %04.3f',n, outputs:mean(), outputs: std(), outputs:max(), outputs:min()))
      end
    end
  end
  self.model:training()

  -- check if bestmodel so far
  local z,bestmodel = self.maskmeter:value('0.7')
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
      self.maskmeter:value('mean'),self.maskmeter:value('median'),
      self.maskmeter:value('0.5'), self.maskmeter:value('0.7'),
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
