require 'nnlr'
require 'nn'



local SC  = nn.SpatialConvolution
local SMP = nn.SpatialMaxPooling
local SAP = nn.SpatialAveragePooling
local RLU = nn.ReLU
local LRN = nn.SpatialCorssMapLRN
local nClasses = 1000



function createModel()
   local modelType = 'E'

   local cfg = {}
   if modelType == 'A' then
      cfg = {64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'B' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'D' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'}
   elseif modelType == 'E' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'}
   else
      error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
   end

   local features = nn.Sequential()
   do
      local iChannels = 3;
      for k,v in ipairs(cfg) do
         if v == 'M' then
            features:add(SMP(2,2,2,2))
         else
            local oChannels = v;
            conv = SC(iChannels,oChannels,3,3,1,1,1,1)
            conv:learningRate('weight', 1)
            conv:weightDecay ('weight', 1)
            conv:learningRate('bias', 2)
            conv:weightDecay ('bias', 0)
            local Nin = conv.nInputPlane * conv.kH * conv.kW
            conv:reset(math.sqrt(1/Nin))
            conv.bias:fill(0.2)

            features:add(conv)
            features:add(RLU(true))
            iChannels = oChannels;
         end
      end
   end

   features:get(1).gradInput = nil

   local classifier = nn.Sequential()
   classifier:add(nn.View(512*7*7))
   classifier:add(nn.Linear(512*7*7, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   --classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   --classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 1000))
   classifier:add(nn.LogSoftMax())

   local model = nn.Sequential()
   model:add(features):add(classifier)

   return model
end



--local module = createModel()
