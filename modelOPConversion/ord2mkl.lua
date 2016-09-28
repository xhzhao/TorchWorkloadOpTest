
local mklOP2ordOP = {}
mklOP2ordOP['nn.SpatialConvolutionMKLDNN'] 	= nn.SpatialConvolution
mklOP2ordOP['nn.SpatialConvolution']	 	= nn.SpatialConvolution
mklOP2ordOP['nn.SpatialMaxPoolingMKLDNN'] 	= nn.SpatialMaxPooling
mklOP2ordOP['nn.SpatialMaxPooling']	 	= nn.SpatialMaxPooling
mklOP2ordOP['nn.SpatialAveragePoolingMKLDNN']	= nn.SpatialAveragePooling
mklOP2ordOP['nn.SpatialAveragePooling']		= nn.SpatialAveragePooling
mklOP2ordOP['nn.LRNMKLDNN'] 			= nn.SpatialCrossMapLRN
mklOP2ordOP['nn.SpatialCrossMapLRN'] 		= nn.SpatialCrossMapLRN
mklOP2ordOP['nn.ReLUMKLDNN'] 			= nn.ReLU
mklOP2ordOP['nn.ReLU']	 			= nn.ReLU
mklOP2ordOP['nn.ConcatMKLDNN'] 			= nn.Concat
mklOP2ordOP['nn.Concat'] 			= nn.Concat
mklOP2ordOP['nn.View'] 				= nn.View


local ordOP2mklOP = {}
ordOP2mklOP['nn.SpatialConvolution'] 		= nn.SpatialConvolutionMKLDNN
ordOP2mklOP['nn.SpatialConvolutionMKLDNN'] 	= nn.SpatialConvolutionMKLDNN
ordOP2mklOP['nn.SpatialMaxPooling']	 	= nn.SpatialMaxPoolingMKLDNN
ordOP2mklOP['nn.SpatialMaxPoolingMKLDNN'] 	= nn.SpatialMaxPoolingMKLDNN
ordOP2mklOP['nn.SpatialAveragePooling']		= nn.SpatialAveragePoolingMKLDNN
ordOP2mklOP['nn.SpatialAveragePoolingMKLDNN']	= nn.SpatialAveragePoolingMKLDNN
ordOP2mklOP['nn.SpatialCrossMapLRN'] 		= nn.LRNMKLDNN
ordOP2mklOP['nn.LRNMKLDNN']		 	= nn.LRNMKLDNN
ordOP2mklOP['nn.ReLU'] 				= nn.ReLUMKLDNN
ordOP2mklOP['nn.ReLUMKLDNN'] 			= nn.ReLUMKLDNN
ordOP2mklOP['nn.Concat'] 			= nn.ConcatMKLDNN
ordOP2mklOP['nn.ConcatMKLDNN'] 			= nn.ConcatMKLDNN
ordOP2mklOP['nn.View']				= nn.View



--[[
NOTE:
the model won't convert to the other version when OPs of source model are same with the refered OPs you specify 
src_model: 	model to be convert to the other version
ori2mkl:   	when ori2mkl==0, the ordinary OP will convert to mkldnn OP
                when ori2mkl!=0, the mkldnn OP will convert to ordinary OP
]]--
function modelTransform(src_model, ori2mkl)
    local cvtOp = (ori2mkl == 0) and ordOP2mklOP or mklOP2ordOP
    return convertModel(src_model, cvtOp)
end

function convertModel(src_module, cvtOP)
    local dst_module
    local module_type = torch.type(src_module)
--    print(module_type)
    if(module_type == 'nn.Sequential') then
	dst_module = nn.Sequential()
	for i = 1, #src_module do
	local src_layer = src_module:get(i)
	local name = src_layer.name
	-- print(name)
	local layer_type = torch.type(src_layer)
	--print(layer_type)
	if(string.find(layer_type, 'SpatialConvolution')) then
        --print('SC')
            local nInputPlane,nOutputPlane = src_layer.nInputPlane, src_layer.nOutputPlane
            local kW,kH = src_layer.kW, src_layer.kH
	    local dW,dH = src_layer.dW, src_layer.dH
	    local padW,padH = src_layer.padW, src_layer.padH
	    local dst_layer = cvtOP[layer_type](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
	    dst_layer.weight:copy(src_layer.weight)
	    dst_layer.bias:copy(src_layer.bias)
	    dst_module:add(dst_layer)
				
	elseif(string.find(layer_type, 'SpatialMaxPooling')) then
        --print('SMP')
	    local kW,kH = src_layer.kW, src_layer.kH
	    local dW,dH = src_layer.dW, src_layer.dH
	    local padW,padH = src_layer.padW, src_layer.padH
	    local dst_layer = cvtOP[layer_type](kW, kH, dW, dH, padW, padH):ceil()
	    dst_module:add(dst_layer)

	elseif(string.find(layer_type, 'SpatialAveragePooling')) then
	--print('SAP')
	    local kW,kH = src_layer.kW, src_layer.kH
	    local dW,dH = src_layer.dW, src_layer.dH
	    local padW,padH = src_layer.padW, src_layer.padH
	    local dst_layer = cvtOP[layer_type](kW, kH, dW, dH, padW, padH)
	    dst_module:add(dst_layer)
				
	elseif(string.find(layer_type, 'LRN')) then
	--print('LRN')
	    local size = src_layer.size
	    local alpha, beta = src_layer.alpha, src_layer.bata
	    local k = src_layer.k
	    local dst_layer = cvtOP[layer_type](size, alpha, beta, k)
	    dst_module:add(dst_layer)

	elseif(string.find(layer_type, 'View')) then
	--print('view')
            local size = src_layer.size
            local dst_layer = cvtOP[layer_type](size):setNumInputDims(3)
            dst_module:add(dst_layer) 
	elseif(string.find(layer_type, 'ReLU')) then
	--print('ReLU')
	    local ip = src_layer.inplace
	    local dst_layer = cvtOP[layer_type](ip)
	    dst_module:add(dst_layer)			
	elseif(string.find(layer_type, 'Concat') or string.find(layer_type, 'Sequential')) then 
            local sub_module = convertModel(src_layer, cvtOP)
            dst_module:add(sub_module)
        else
            local new_layer = src_layer:clone()
            dst_module:add(new_layer)
	end
		end
    elseif(string.find(module_type, 'Concat')) then
	local dimension = src_module.dimension
	dst_module = nn.Concat(dimension)
	for j = 1, src_module:size() do 
            local dnn = src_module:get(j)
	    local sub_module = convertModel(dnn, cvtOP)
            dst_module:add(sub_module)
	end
        
    end
    --print(dst_module)
    return dst_module	
end
