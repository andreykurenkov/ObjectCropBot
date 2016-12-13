--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Train DeepMask or SharpMask
------------------------------------------------------------------------------]]

require 'torch'
require 'cutorch'
require 'cudnn'
require 'nn'
--------------------------------------------------------------------------------
-- parse arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('train DeepMask or SharpMask')
cmd:text()
cmd:text('Options:')
cmd:option('-rundir', 'exps/', 'experiments directory')
cmd:option('-name', '', 'name of experiment')
cmd:option('-datadir', 'data/', 'data directory')
cmd:option('-seed', 1, 'manually set RNG seed')
cmd:option('-gpu', 1, 'gpu device')
cmd:option('-nthreads', 4, 'number of threads for DataSampler')
cmd:option('-reload', '', 'reload a network from given directory')
cmd:option('-preload', '', 'train DeepCrop with starting weights from DeepMask')
cmd:text()
cmd:text('Training Options:')
cmd:option('-batch', 16, 'training batch size')
cmd:option('-lr', 0, 'learning rate (0 uses default lr schedule)')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('-wd', 5e-4, 'weight decay')
cmd:option('-maxload', 4000, 'max number of training batches per epoch')
cmd:option('-testmaxload', 250, 'max number of testing batches')
cmd:option('-maxepoch', 200, 'max number of training epochs')
cmd:option('-shift', 16, 'shift jitter allowed')
cmd:option('-scale', .25, 'scale jitter allowed')
cmd:option('-hfreq', 0.5, 'mask/score head sampling frequency')
cmd:option('-iSz', 160, 'input size')
cmd:option('-oSz', 56, 'output size')
cmd:option('-gSz', 112, 'ground truth size')
cmd:option('-resnet50', false, 'Whether to train with resnet-50 instead of resnet-18')
cmd:option('-scratch', false, 'train DeepCrop with randomly initialize weights')
cmd:option('-verbose', true, 'train DeepCrop with extra output')
cmd:text()
cmd:text('SharpMask Options:')
cmd:option('-dm', '', 'path to trained deepmask (if dm, then train SharpMask)')
cmd:option('-km', 32, 'km')
cmd:option('-ks', 32, 'ks')

local config = cmd:parse(arg)

--------------------------------------------------------------------------------
-- various initializations
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(config.gpu)
torch.manualSeed(config.seed)
math.randomseed(config.seed)

local trainSm -- flag to train SharpMask (true) or DeepMask (false)
if #config.dm > 0 then
  trainSm = true
  config.hfreq = 0 -- train only mask head
  config.gSz = config.iSz -- in sharpmask, ground-truth has same dim as input
end

paths.dofile('SpatialSymmetricPadding.lua')
paths.dofile('DeepMask.lua')
paths.dofile('DeepCrop.lua')
if trainSm then paths.dofile('SharpMask.lua') end

--------------------------------------------------------------------------------
-- reload?
local epoch, model
if #config.reload > 0 then
  epoch = 0
  if paths.filep(config.reload..'/log') then
    for line in io.lines(config.reload..'/log') do
      if string.find(line,'train') then epoch = epoch + 1 end
    end
  end
  print(string.format('| reloading experiment %s', config.reload))
  local m = torch.load(string.format('%s/model.t7', config.reload))
  model, config = m.model, m.config
elseif #config.preload > 0 then
  print(string.format('| reloading DeepMask experiment %s', config.preload))
  local m = torch.load(string.format('%s/model.t7', config.preload))
  local maskModel = m.model

  local inLayer = nn.SpatialConvolution(4, 64, 7, 7, 2,2):cuda()
  inLayer.weight[{ 2,{1,3} }]:set(maskModel.trunk.modules[1].modules[2].weight)
  maskModel.trunk.modules[1].modules[2] = inLayer
  --maskModel.trunk.modules[11]=nn.View(config.batch,128*10*10):cuda()
  config.resnet50=true
  model = nn.DeepCrop(config)
  model.trunk=maskModel.trunk
  model.maskBranch=maskModel.maskBranch
  model.scoreBranch=maskModel.scoreBranch
end

--------------------------------------------------------------------------------
-- directory to save log and model
local pathsv = trainSm and 'sharpmask/exp' or 'deepcrop/exp'
config.rundir = cmd:string(
  paths.concat(config.reload=='' and config.rundir or config.reload, pathsv),
  config,{rundir=true, gpu=true, reload=true, datadir=true, dm=true} --ignore
)

print(string.format('| running in directory %s', config.rundir))
os.execute(string.format('mkdir -p %s',config.rundir))
os.execute(string.format('mkdir -p %s/samples/train',config.rundir))
os.execute(string.format('mkdir -p %s/samples/test',config.rundir))

--------------------------------------------------------------------------------
-- network and criterion
model = model or (trainSm and nn.SharpCrop(config) or nn.DeepCrop(config))
local criterion = nn.SoftMarginCriterion():cuda()

print('| start training')

--------------------------------------------------------------------------------
-- initialize data loader
local DataLoader = paths.dofile('DataLoader.lua')
local trainLoader, valLoader = DataLoader.create(config)

--------------------------------------------------------------------------------
-- initialize Trainer (handles training/testing loop)
if trainSm then
  paths.dofile('TrainerSharpMask.lua')
else
  paths.dofile('TrainerDeepCrop.lua')
end
local trainer = Trainer(model, criterion, config)
--------------------------------------------------------------------------------
-- do it
local trainLossStr = '1'
local trainErrorStr = '1'
local testErrorStr = '1'

epoch = epoch or 1
for i = 1, config.maxepoch do
  trainer:train(epoch,trainLoader)

  trainLossStr = string.format('%s,%f',trainLossStr,trainer.lossmeter:value())

  if config.verbose then
     print('| Train loss:')
     print(trainLossStr)
  end

  if i%2 == 0 then 
    trainer:test(epoch,valLoader) 

    testErrorStr = string.format('%s,%f',testErrorStr,1-trainer.maskmeter:value('0.7'))

    if config.verbose then
      print('| Test error:')
      print(trainErrorStr)
    end
  end

  epoch = epoch + 1
end
print('| training finished')
print('| Train loss:')
print(trainLossStr)
print('| Test error:')
print(testErrorStr)
