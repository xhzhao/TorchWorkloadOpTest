require 'paths'
require 'nn'
require 'mklnn'

paths.dofile('alexnet.lua')  -- corresponding file must provide a function named as createModel() to create your expected MKLDNN  model


torch.setdefaulttensortype('torch.FloatTensor')

local ori_model = createModel()     -- build the model


		
local dnnprimitives = torch.LongTensor(2)

local batch = 2 --math.random(3,8)
--local channel, ini, inj= 256,6,6  --linear input test
local channel, ini, inj= 3,227,227  --full model test
--local channel, ini, inj= 256,13,13  --maxpool + linear test
--local channel, ini, inj= 384,13,13  --conv + maxpool + linear test

local input = torch.randn(batch, channel, inj, ini)
--local input = torch.randn(batch, 256, 6, 6)
local input_clone
local gradOutput
--local ori_model = copyOriModule(dnn_model)  --convert MKLDNN OP module to ordinary OP model
local gradOutput_clone
local dnnOutput
--local ori_model
local dnn_model = mklnn.convert(ori_model, 'mkl')
print("dnn_model = ",dnn_model)

for i=1,1000  do
   --collectgarbage()
   dnn_model:zeroGradParameters()
   ori_model:zeroGradParameters()
-------forward
   input = torch.randn(batch, channel, inj, ini):uniform(0,255)
   input_clone = input:clone()
   input = input:mkl()
   dnnOutput=dnn_model:forward(input)
   ori_model:forward(input_clone)

-------backward
   gradOutput = torch.randn(dnnOutput:size()):uniform(-0.01,0)
   gradOutput_clone = gradOutput:clone()
   dnn_model:backward(input, gradOutput)
   ori_model:backward(input_clone, gradOutput_clone)


   local param, gradParam = dnn_model:getParameters()
   local oriparam, origradParam = ori_model:getParameters()
   print("dnn gradParam sum = ", gradParam:sum(), ", origradParam sum = ",origradParam:sum(),", size = ", gradParam:nElement())


end
