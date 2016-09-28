require 'paths'

paths.dofile('alexnet_mkldnn.lua')  -- corresponding file must provide a function named as createModel() to create your expected MKLDNN  model
--paths.dofile('main0_module.lua')  -- corresponding file must provide a function named as createModel() to create your expected MKLDNN  model

--main0_module.lua

torch.setdefaulttensortype('torch.FloatTensor')

local dnn_model = createModel()     -- build the model


function copyOriModule(dnn_module)
    local ori_module
	local module_type = torch.type(dnn_module)
--    print(module_type)
	if(module_type == 'nn.Sequential') then
		ori_module = nn.Sequential()
		for i = 1, #dnn_module do
			local dnn_layer = dnn_module:get(i)
			local name = dnn_layer.name
		   -- print(name)
			local layer_type = torch.type(dnn_layer)
			--print(layer_type)
			if(layer_type == 'nn.SpatialConvolutionMKLDNN') then

          --      print('SC')
                local nInputPlane,nOutputPlane = dnn_layer.nInputPlane, dnn_layer.nOutputPlane
                local kW,kH = dnn_layer.kW, dnn_layer.kH
				local dW,dH = dnn_layer.dW, dnn_layer.dH
				local padW,padH = dnn_layer.padW, dnn_layer.padH
				local ori_layer = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
        --        print('copy')
				ori_layer.weight:copy(dnn_layer.weight)
				ori_layer.bias:copy(dnn_layer.bias)
				ori_module:add(ori_layer)
				
			elseif(layer_type == 'nn.SpatialMaxPoolingMKLDNN') then
      --          print('SMP')
				local kW,kH = dnn_layer.kW, dnn_layer.kH
				local dW,dH = dnn_layer.dW, dnn_layer.dH
				local padW,padH = dnn_layer.padW, dnn_layer.padH
				local ori_layer = nn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH):ceil()
				ori_module:add(ori_layer)
				
			elseif(layer_type == 'nn.SpatialAveragePoolingMKLDNN') then
    --            print('SAP')
				local kW,kH = dnn_layer.kW, dnn_layer.kH
				local dW,dH = dnn_layer.dW, dnn_layer.dH
				local padW,padH = dnn_layer.padW, dnn_layer.padH
				local ori_layer = nn.SpatialAveragePooling(kW, kH, dW, dH, padW, padH)
				ori_module:add(ori_layer)
				
			elseif(layer_type == 'nn.LRNMKLDNN') then
  --              print('LRN')
				local size = dnn_layer.size
				local alpha, beta = dnn_layer.alpha, dnn_layer.bata
				local k = dnn_layer.k
				local ori_layer = nn.SpatialCrossMapLRN(size, alpha, beta, k)
				ori_module:add(ori_layer)

			elseif(layer_type == 'nn.View') then
                --print('    ')
                --print('view')
                local size = dnn_layer.size
                --print(size)
                local ori_layer = nn.View(size):setNumInputDims(3)
                ori_module:add(ori_layer) 
			elseif(layer_type == 'nn.ReLUMKLDNN') then
--                print('ReLU')
				local ip = dnn_layer.inplace
				local ori_layer = nn.ReLU(ip)
				ori_module:add(ori_layer)			
            elseif((layer_type == 'nn.ConcatMKLDNN') or (layer_type == 'nn.Concat') or (layer_type == 'nn.Sequential')) then 
       --         print(layer_type)
--                print(dnn_layer)
                local sub_module = copyOriModule(dnn_layer)
--                print(sub_module)
                ori_module:add(sub_module)
            else
                --print(layer_type)
                local new_layer = dnn_layer:clone()
                --print(dnn_layer)
                ori_module:add(new_layer)

            end
		end
	elseif((module_type == 'nn.ConcatMKLDNN') or (module_type == 'nn.Concat')) then
		local dimension = dnn_module.dimension
		ori_module = nn.Concat(dimension)
		for j = 1, dnn_module:size() do 
            local dnn = dnn_module:get(j)
           
			local sub_module = copyOriModule(dnn)
          --  print('sub_module')
           -- print(sub_module)
            ori_module:add(sub_module)
		end
        
	end
    --print(ori_module)
    return ori_module	
end
		
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
local ori_model = copyOriModule(dnn_model)  --convert MKLDNN OP module to ordinary OP model

for i=1,1000  do
   --collectgarbage()
   dnn_model:zeroGradParameters()
   ori_model:zeroGradParameters()
-------forward
   input = torch.randn(batch, channel, inj, ini):uniform(0,255)
   input_clone = input:clone()
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
