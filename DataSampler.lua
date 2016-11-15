--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Dataset sampler for for training/evaluation of DeepMask and SharpMask
------------------------------------------------------------------------------]]

require 'torch'
require 'image'
local tds = require 'tds'
local coco = require 'coco'

local DataSampler = torch.class('DataSampler')

--------------------------------------------------------------------------------
-- function: init
function DataSampler:__init(config,split)
  assert(split == 'train' or split == 'val')

  -- coco api
  local annFile = string.format('%s/annotations/instances_%s2014.json',
  config.datadir,split)
  self.coco = coco.CocoApi(annFile)

  -- mask api
  self.maskApi = coco.MaskApi

  -- mean/std computed from random subset of ImageNet training images
  self.mean, self.std = {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}

  -- class members
  self.datadir = config.datadir
  self.split = split

  self.iSz = config.iSz
  self.objSz = math.ceil(config.iSz*128/224)
  self.wSz = config.iSz + 32
  self.gSz = config.gSz
  self.scale = config.scale
  self.shift = config.shift

  self.imgIds = self.coco:getImgIds()
  self.annIds = self.coco:getAnnIds()
  self.catIds = self.coco:getCatIds()
  self.nImages = self.imgIds:size(1)

  --self.imgWindow = image.window()
  --self.lblWindow = image.window()
  --self.distWindow = image.window()

  if split == 'train' then self.__size  = config.maxload*config.batch
  elseif split == 'val' then self.__size = config.testmaxload*config.batch end

  collectgarbage()
end
local function log2(x) return math.log(x)/math.log(2) end

--------------------------------------------------------------------------------
-- function: get size of epoch
function DataSampler:size()
  return self.__size
end

--------------------------------------------------------------------------------
-- function: get a sample
function DataSampler:get()
  local input,label
  input, label = self:maskSampling()

  if input==nil then
      return nil,nil
  end
  -- normalize input
  for i=1,3 do input:select(4,1):narrow(1,i,1):add(-self.mean[i]):div(self.std[i]) end
  --for i=1,3 do input:select(4,2):narrow(1,i,1):add(-self.mean[i]):div(self.std[i]) end

  return input,label
end

--------------------------------------------------------------------------------
-- function: mask sampling
function DataSampler:maskSampling()
  local iSz,wSz,gSz = self.iSz,self.wSz,self.gSz

  local cat,ann = torch.random(80)
  while not ann or ann.iscrowd == 1 or ann.area < 100 or ann.bbox[3] < 5
    or ann.bbox[4] < 5 do
      local catId = self.catIds[cat]
      local annIds = self.coco:getAnnIds({catId=catId})
      local annid = annIds[torch.random(annIds:size(1))]
      ann = self.coco:loadAnns(annid)[1]
  end
  local bbox = ann.bbox --self:jitterBox(ann.bbox)
  local imgName = self.coco:loadImgs(ann.image_id)[1].file_name
  -- input
  local pathImg = string.format('%s/%s2014/%s',self.datadir,self.split,imgName)
  local inp = image.load(pathImg,3)
  local h, w = inp:size(2), inp:size(3)
  -- inp = self:cropTensor(inp, bbox, 0.5)
  local imgInp = image.scale(inp, wSz, wSz)
  --image.display{input=imgInp,gui=false,window=self.imgWindow}

  -- label
  local iSzR = iSz*(bbox[3]/wSz)
  local xc, yc = bbox[1]+bbox[3]/2, bbox[2]+bbox[4]/2
  local bboxInpSz = {xc-iSzR/2,yc-iSzR/2,iSzR,iSzR}
  local lbl = self:cropMask(ann, bboxInpSz, h, w, gSz)
  lbl:mul(2):add(-1)
  --image.display{input=scaledLbl,gui=false,window=self.lblWindow}

  local imgInp, distanceInp = self:calcDistanceInp(imgInp, lbl, gSz, wSz)
  if distanceInp == nil then
      return nil, nil
  end
  --Create combine 3 x wSz x wSz input x 2
  local combinedInp = torch.cat(imgInp,distanceInp,4)
  return combinedInp, lbl
end


--------------------------------------------------------------------------------
-- function: generate 'click' inside bounded object and get additional inputs for it
function DataSampler:calcDistanceInp(imgInp, lbl, gSz, wSz)
  local distanceInp = torch.FloatTensor(3,wSz,wSz)

  -- Sample a 'crop click' pixel
  local cropClick
  count = lbl:gt(0):sum()
  if count==1 or count<(wSz/8) then
    --- Skip samples with no or very small crop borders
    return nil, nil
  else
    flatLbl = lbl:reshape(gSz^2)
    idx = torch.linspace(0,gSz^2-1,gSz^2)[flatLbl:gt(0)]
    cropClick = math.random(count)
  end
  clickIdx = math.floor(idx[cropClick])
  cropClickX = math.floor((clickIdx%gSz+1)*wSz/gSz)
  cropClickY = math.floor(math.floor(clickIdx/gSz+1)*wSz/gSz)
  imgInp[{{1,3},cropClickX,cropClickY}] = torch.Tensor({-1,-1,-1})

  -- Calculate location difference from click pixel, via 2 norm
  pixels = torch.Tensor(torch.linspace(1,wSz,wSz))
  pixelsX = pixels:reshape(wSz,1):repeatTensor(1,wSz)
  pixelsY = pixels:reshape(1,wSz):repeatTensor(wSz,1)
  coords = pixelsX:cat(pixelsY,3):transpose(1,3)
  clickXs = torch.Tensor({cropClickX}):repeatTensor(wSz,wSz)
  clickYs = torch.Tensor({cropClickY}):repeatTensor(wSz,wSz)
  clickCoords = clickXs:cat(clickYs,3):transpose(1,3)
  dists = (coords-clickCoords):norm(2,1)
  dists = (dists:div(dists:max())-0.5)*2
  distanceInp[1] = dists

  -- Calculate rgb difference from click pixel, via 2 nom
  local pixel = imgInp[{{1,3},cropClickY,cropClickX}]
  pixels = pixel:reshape(1,1,3):repeatTensor(wSz,wSz,1):transpose(1,3)
  distanceInp[2] = (imgInp-pixels):norm(2,1)
  
  -- Calculate lum difference from click pixel, via 2 nom
  lumTensor = torch.Tensor({0.299,0.587,0.114}):reshape(3,1,1)
  imgLum = imgInp:conv3(lumTensor)
  pixelLum = pixels:conv3(lumTensor)
  distanceInp[3] = (imgLum-pixelLum):norm(2,1)

  return imgInp,distanceInp
end


--------------------------------------------------------------------------------
-- function: crop bbox b from inp tensor
function DataSampler:cropTensor(inp, b, pad)
  pad = pad or 0
  b[1], b[2] = torch.round(b[1])+1, torch.round(b[2])+1 -- 0 to 1 index
  b[3], b[4] = torch.round(b[3]), torch.round(b[4])

  local out, h, w, ind
  if #inp:size() == 3 then
    ind, out = 2, torch.Tensor(inp:size(1), b[3], b[4]):fill(pad)
  elseif #inp:size() == 2 then
    ind, out = 1, torch.Tensor(b[3], b[4]):fill(pad)
  end
  h, w = inp:size(ind), inp:size(ind+1)

  local xo1,yo1,xo2,yo2 = b[1],b[2],b[3]+b[1]-1,b[4]+b[2]-1
  local xc1,yc1,xc2,yc2 = 1,1,b[3],b[4]

  -- compute box on binary mask inp and cropped mask out
  if b[1] < 1 then xo1=1; xc1=1+(1-b[1]) end
  if b[2] < 1 then yo1=1; yc1=1+(1-b[2]) end
  if b[1]+b[3]-1 > w then xo2=w; xc2=xc2-(b[1]+b[3]-1-w) end
  if b[2]+b[4]-1 > h then yo2=h; yc2=yc2-(b[2]+b[4]-1-h) end
  local xo, yo, wo, ho = xo1, yo1, xo2-xo1+1, yo2-yo1+1
  local xc, yc, wc, hc = xc1, yc1, xc2-xc1+1, yc2-yc1+1
  if yc+hc-1 > out:size(ind)   then hc = out:size(ind  )-yc+1 end
  if xc+wc-1 > out:size(ind+1) then wc = out:size(ind+1)-xc+1 end
  if yo+ho-1 > inp:size(ind)   then ho = inp:size(ind  )-yo+1 end
  if xo+wo-1 > inp:size(ind+1) then wo = inp:size(ind+1)-xo+1 end
  out:narrow(ind,yc,hc); out:narrow(ind+1,xc,wc)
  inp:narrow(ind,yo,ho); inp:narrow(ind+1,xo,wo)
  out:narrow(ind,yc,hc):narrow(ind+1,xc,wc):copy(
  inp:narrow(ind,yo,ho):narrow(ind+1,xo,wo))

  return out
end

--------------------------------------------------------------------------------
-- function: crop bbox from mask
function DataSampler:cropMask(ann, bbox, h, w, sz)
  local mask = torch.FloatTensor(sz,sz)
  local seg = ann.segmentation

  local scale = sz / bbox[3]
  local polS = {}
  for m, segm in pairs(seg) do
    polS[m] = torch.DoubleTensor():resizeAs(segm):copy(segm); polS[m]:mul(scale)
  end
  --local bboxS = {}
  --for m = 1,#bbox do bboxS[m] = bbox[m]*scale end

  local Rs = self.maskApi.frPoly(polS, h*scale, w*scale)
  local mo = self.maskApi.decode(Rs)
  ---local mc = self:cropTensor(mo, bboxS)
  mask:copy(image.scale(mo,sz,sz):gt(0.5))

  return mask
end

return DataSampler
