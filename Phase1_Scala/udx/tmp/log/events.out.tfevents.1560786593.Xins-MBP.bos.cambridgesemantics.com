       �K"	��e��A�Abrain.Event:2 ���`     ����	X9d��A�AJ��
*
1.12.0-rc0��
s
global_epochVarHandleOp*
dtype0	*
	container *
shape: *
shared_nameglobal_epoch*
_class
 
�
Hglobal_epoch/Initializer/global_epoch/Initializer/ZerosInitializer/ShapeConst*
valueB *
_class
loc:@global_epoch*
dtype0
�
Qglobal_epoch/Initializer/global_epoch/Initializer/ZerosInitializer/Zeros/ConstantConst*
valueB	 "        *
_class
loc:@global_epoch*
dtype0	
�
Mglobal_epoch/Initializer/global_epoch/Initializer/ZerosInitializer/Zeros/FillFillHglobal_epoch/Initializer/global_epoch/Initializer/ZerosInitializer/ShapeQglobal_epoch/Initializer/global_epoch/Initializer/ZerosInitializer/Zeros/Constant*

index_type0*
_class
loc:@global_epoch*
T0	
�
!global_epoch/InitializationAssignAssignVariableOpglobal_epochMglobal_epoch/Initializer/global_epoch/Initializer/ZerosInitializer/Zeros/Fill*
dtype0	
l
global_epoch/Read/ReadVariableReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
>
global_epoch/IsInitializedVarIsInitializedOpglobal_epoch
t
&global_epoch/global_epoch/ReadVariableReadVariableOpglobal_epoch*
dtype0	*
_class
loc:@global_epoch
q
global_stepVarHandleOp*
_class
 *
dtype0	*
	container *
shape: *
shared_nameglobal_step
�
Fglobal_step/Initializer/global_step/Initializer/ZerosInitializer/ShapeConst*
valueB *
_class
loc:@global_step*
dtype0
�
Oglobal_step/Initializer/global_step/Initializer/ZerosInitializer/Zeros/ConstantConst*
valueB	 "        *
_class
loc:@global_step*
dtype0	
�
Kglobal_step/Initializer/global_step/Initializer/ZerosInitializer/Zeros/FillFillFglobal_step/Initializer/global_step/Initializer/ZerosInitializer/ShapeOglobal_step/Initializer/global_step/Initializer/ZerosInitializer/Zeros/Constant*
T0	*

index_type0*
_class
loc:@global_step
�
 global_step/InitializationAssignAssignVariableOpglobal_stepKglobal_step/Initializer/global_step/Initializer/ZerosInitializer/Zeros/Fill*
dtype0	
i
global_step/Read/ReadVariableReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
<
global_step/IsInitializedVarIsInitializedOpglobal_step
p
$global_step/global_step/ReadVariableReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
�
Estimator/Train/Model/IteratorIterator*
output_types
2*
shared_name *5
output_shapes$
":���������:���������*
	container 
�
*Estimator/Train/Model/Input_Input/Zip/NextIteratorGetNextEstimator/Train/Model/Iterator*5
output_shapes$
":���������:���������*
output_types
2
P
Estimator/Train/Model/ShapeConst*
valueB"����   *
dtype0
�
#Estimator/Train/Model/Input/FlattenReshape*Estimator/Train/Model/Input_Input/Zip/NextEstimator/Train/Model/Shape*
T0*
Tshape0
�
\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/WeightsVarHandleOp*
_class
 *
dtype0*
	container *
shape
:@*m
shared_name^\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/ShapeConst*
valueB"   @   *o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/ConstantConst*
valueB
 "    *o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Constant_1Const*
valueB
 "  �?*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/RandomNormalInitializerRandomStandardNormal�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Shape*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0*
seed2 *

seed *
T0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/MultiplyMul�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/RandomNormalInitializer�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Constant_1*
T0*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/AddAdd�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Multiply�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Constant*
T0*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights
�
qInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/InitializationAssignAssignVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Add*
dtype0
�
nInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Read/ReadVariableReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
jInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/IsInitializedVarIsInitializedOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/ReadVariableReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
"Estimator/Train/Model/ReadVariableReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
YInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/BiasVarHandleOp*
	container *
shape:@*j
shared_name[YInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
_class
 *
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/ShapeConst*
valueB"@   *l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/ConstantConst*
dtype0*
valueB
 "    *l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Constant_1Const*
valueB
 "  �?*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/RandomNormalInitializerRandomStandardNormal�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Shape*
dtype0*
seed2 *

seed *
T0*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/MultiplyMul�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/RandomNormalInitializer�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Constant_1*
T0*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/AddAdd�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Multiply�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Constant*
T0*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias
�
nInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/InitializationAssignAssignVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Add*
dtype0
�
kInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Read/ReadVariableReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
gInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/IsInitializedVarIsInitializedOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/ReadVariableReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
$Estimator/Train/Model/ReadVariable_1ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
#Estimator/Train/Model/Linear/MatMulMatMul#Estimator/Train/Model/Input/Flatten"Estimator/Train/Model/ReadVariable*
transpose_a( *
transpose_b( *
T0
�
$Estimator/Train/Model/Linear/AddBiasBiasAdd#Estimator/Train/Model/Linear/MatMul$Estimator/Train/Model/ReadVariable_1*
T0*
data_formatNHWC
c
,Estimator/Train/Model/ReLU/ReLU/PositivePartRelu$Estimator/Train/Model/Linear/AddBias*
T0
W
!Estimator/Train/Model/ReLU/NegateNeg$Estimator/Train/Model/Linear/AddBias*
T0
`
,Estimator/Train/Model/ReLU/ReLU/NegativePartRelu!Estimator/Train/Model/ReLU/Negate*
T0
P
#Estimator/Train/Model/ReLU/ConstantConst*
valueB
 "���=*
dtype0
�
#Estimator/Train/Model/ReLU/MultiplyMul#Estimator/Train/Model/ReLU/Constant,Estimator/Train/Model/ReLU/ReLU/NegativePart*
T0
�
#Estimator/Train/Model/ReLU/SubtractSub,Estimator/Train/Model/ReLU/ReLU/PositivePart#Estimator/Train/Model/ReLU/Multiply*
T0
�
@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/WeightsVarHandleOp*
_class
 *
dtype0*
	container *
shape
:@@*Q
shared_nameB@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/ShapeConst*
valueB"@   @   *S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/ConstantConst*
valueB
 "    *S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Constant_1Const*
valueB
 "  �?*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/RandomNormalInitializerRandomStandardNormal�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Shape*
T0*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0*
seed2 *

seed 
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/MultiplyMul�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/RandomNormalInitializer�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Constant_1*
T0*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/AddAdd�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Multiply�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Constant*
T0*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights
�
UInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/InitializationAssignAssignVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Add*
dtype0
�
RInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Read/ReadVariableReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
NInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/IsInitializedVarIsInitializedOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/ReadVariableReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
$Estimator/Train/Model/ReadVariable_2ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/BiasVarHandleOp*N
shared_name?=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
_class
 *
dtype0*
	container *
shape:@
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/ShapeConst*
valueB"@   *P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/ConstantConst*
valueB
 "    *P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Constant_1Const*
valueB
 "  �?*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/RandomNormalInitializerRandomStandardNormal�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Shape*

seed *
T0*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0*
seed2 
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/MultiplyMul�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/RandomNormalInitializer�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Constant_1*
T0*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/AddAdd�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Multiply�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Constant*
T0*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias
�
RInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/InitializationAssignAssignVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Add*
dtype0
�
OInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Read/ReadVariableReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
KInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/IsInitializedVarIsInitializedOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/ReadVariableReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
$Estimator/Train/Model/ReadVariable_3ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
%Estimator/Train/Model/Linear_1/MatMulMatMul#Estimator/Train/Model/ReLU/Subtract$Estimator/Train/Model/ReadVariable_2*
T0*
transpose_a( *
transpose_b( 
�
&Estimator/Train/Model/Linear_1/AddBiasBiasAdd%Estimator/Train/Model/Linear_1/MatMul$Estimator/Train/Model/ReadVariable_3*
T0*
data_formatNHWC
g
.Estimator/Train/Model/ReLU_1/ReLU/PositivePartRelu&Estimator/Train/Model/Linear_1/AddBias*
T0
[
#Estimator/Train/Model/ReLU_1/NegateNeg&Estimator/Train/Model/Linear_1/AddBias*
T0
d
.Estimator/Train/Model/ReLU_1/ReLU/NegativePartRelu#Estimator/Train/Model/ReLU_1/Negate*
T0
R
%Estimator/Train/Model/ReLU_1/ConstantConst*
valueB
 "���=*
dtype0
�
%Estimator/Train/Model/ReLU_1/MultiplyMul%Estimator/Train/Model/ReLU_1/Constant.Estimator/Train/Model/ReLU_1/ReLU/NegativePart*
T0
�
%Estimator/Train/Model/ReLU_1/SubtractSub.Estimator/Train/Model/ReLU_1/ReLU/PositivePart%Estimator/Train/Model/ReLU_1/Multiply*
T0
�
(Input/Flatten/OutputLayer/Linear/WeightsVarHandleOp*
dtype0*
	container *
shape
:@*9
shared_name*(Input/Flatten/OutputLayer/Linear/Weights*
_class
 
�
oInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/ShapeConst*
valueB"@      *;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
rInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/ConstantConst*
valueB
 "    *;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
tInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/Constant_1Const*
valueB
 "  �?*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
�Input/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/RandomNormalInitializerRandomStandardNormaloInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/Shape*
seed2 *

seed *
T0*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
rInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/MultiplyMul�Input/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/RandomNormalInitializertInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/Constant_1*
T0*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights
�
mInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/AddAddrInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/MultiplyrInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/Constant*
T0*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights
�
=Input/Flatten/OutputLayer/Linear/Weights/InitializationAssignAssignVariableOp(Input/Flatten/OutputLayer/Linear/WeightsmInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/Add*
dtype0
�
:Input/Flatten/OutputLayer/Linear/Weights/Read/ReadVariableReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
v
6Input/Flatten/OutputLayer/Linear/Weights/IsInitializedVarIsInitializedOp(Input/Flatten/OutputLayer/Linear/Weights
�
^Input/Flatten/OutputLayer/Linear/Weights/Input/Flatten/OutputLayer/Linear/Weights/ReadVariableReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
$Estimator/Train/Model/ReadVariable_4ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
%Input/Flatten/OutputLayer/Linear/BiasVarHandleOp*
shape:*6
shared_name'%Input/Flatten/OutputLayer/Linear/Bias*
_class
 *
dtype0*
	container 
�
iInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/ShapeConst*
valueB"   *8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
lInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/ConstantConst*
valueB
 "    *8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
nInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/Constant_1Const*
valueB
 "  �?*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
{Input/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/RandomNormalInitializerRandomStandardNormaliInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/Shape*
seed2 *

seed *
T0*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
lInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/MultiplyMul{Input/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/RandomNormalInitializernInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/Constant_1*
T0*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias
�
gInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/AddAddlInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/MultiplylInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/Constant*
T0*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias
�
:Input/Flatten/OutputLayer/Linear/Bias/InitializationAssignAssignVariableOp%Input/Flatten/OutputLayer/Linear/BiasgInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/Add*
dtype0
�
7Input/Flatten/OutputLayer/Linear/Bias/Read/ReadVariableReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
p
3Input/Flatten/OutputLayer/Linear/Bias/IsInitializedVarIsInitializedOp%Input/Flatten/OutputLayer/Linear/Bias
�
XInput/Flatten/OutputLayer/Linear/Bias/Input/Flatten/OutputLayer/Linear/Bias/ReadVariableReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
$Estimator/Train/Model/ReadVariable_5ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*
dtype0*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias
�
%Estimator/Train/Model/Linear_2/MatMulMatMul%Estimator/Train/Model/ReLU_1/Subtract$Estimator/Train/Model/ReadVariable_4*
transpose_a( *
transpose_b( *
T0
�
&Estimator/Train/Model/Linear_2/AddBiasBiasAdd%Estimator/Train/Model/Linear_2/MatMul$Estimator/Train/Model/ReadVariable_5*
T0*
data_formatNHWC
�
Estimator/Train/Model/SubtractSub&Estimator/Train/Model/Linear_2/AddBias,Estimator/Train/Model/Input_Input/Zip/Next:1*
T0
P
Estimator/Train/Model/Loss/L2L2LossEstimator/Train/Model/Subtract*
T0
Z
1Estimator/Train/Model/Gradients/Gradients_0/ShapeConst*
valueB *
dtype0
f
9Estimator/Train/Model/Gradients/Gradients_0/Ones/ConstantConst*
valueB
 "  �?*
dtype0
�
5Estimator/Train/Model/Gradients/Gradients_0/Ones/FillFill1Estimator/Train/Model/Gradients/Gradients_0/Shape9Estimator/Train/Model/Gradients/Gradients_0/Ones/Constant*
T0*

index_type0
�
NEstimator/Train/Model/Gradients/Estimator/Train/Model/Loss/L2Gradient/MultiplyMulEstimator/Train/Model/Subtract5Estimator/Train/Model/Gradients/Gradients_0/Ones/Fill*
T0
�
LEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/ShapeShape&Estimator/Train/Model/Linear_2/AddBias*
T0*
out_type0
�
NEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Shape_1Shape,Estimator/Train/Model/Input_Input/Zip/Next:1*
T0*
out_type0
�
aEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/BroadcastGradientArgumentsBroadcastGradientArgsLEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/ShapeNEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Shape_1*
T0
�
JEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/SumSumNEstimator/Train/Model/Gradients/Estimator/Train/Model/Loss/L2Gradient/MultiplyaEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/BroadcastGradientArguments*
T0*

Tidx0*
	keep_dims( 
�
NEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/ReshapeReshapeJEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/SumLEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Shape*
T0*
Tshape0*
_class
 
�
LEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Sum_1SumNEstimator/Train/Model/Gradients/Estimator/Train/Model/Loss/L2Gradient/MultiplycEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/BroadcastGradientArguments:1*

Tidx0*
	keep_dims( *
T0
�
MEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/NegateNegLEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Sum_1*
T0
�
PEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Reshape_1ReshapeMEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/NegateNEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Shape_1*
T0*
_class
 *
Tshape0
�
REstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/GroupNoOpO^Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/ReshapeQ^Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Reshape_1
�
vEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityNEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/ReshapeS^Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/Group*
T0*a
_classW
USloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Reshape
�
xEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityPEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Reshape_1S^Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/Group*
T0*c
_classY
WUloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Reshape_1
�
^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/BiasAddGradientBiasAddGradvEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/WithControlDependencies/Identity/Identity*
T0*
_class
 *
data_formatNHWC
�
ZEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/GroupNoOp_^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/BiasAddGradientw^Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/WithControlDependencies/Identity/Identity
�
~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityvEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/WithControlDependencies/Identity/Identity[^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/Group*
T0*�
_class�
�Sloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Reshape{loc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/WithControlDependencies/Identity/Identity
�
�Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentity^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/BiasAddGradient[^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/Group*
T0*q
_classg
ecloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/BiasAddGradient
�
TEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMulMatMul~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity$Estimator/Train/Model/ReadVariable_4*
T0*
_class
 *
transpose_a( *
transpose_b(
�
VEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul_1MatMul%Estimator/Train/Model/ReLU_1/Subtract~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity*
transpose_b( *
T0*
_class
 *
transpose_a(
�
YEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/GroupNoOpU^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMulW^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul_1
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityTEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMulZ^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/Group*
T0*g
_class]
[Yloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul
�
Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityVEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul_1Z^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/Group*
T0*i
_class_
][loc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul_1
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/ShapeShape.Estimator/Train/Model/ReLU_1/ReLU/PositivePart*
T0*
out_type0
�
UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Shape_1Shape%Estimator/Train/Model/ReLU_1/Multiply*
T0*
out_type0
�
hEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/BroadcastGradientArgumentsBroadcastGradientArgsSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/ShapeUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Shape_1*
T0
�
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/SumSum}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityhEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/BroadcastGradientArguments*

Tidx0*
	keep_dims( *
T0
�
UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/ReshapeReshapeQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/SumSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Shape*
T0*
_class
 *
Tshape0
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Sum_1Sum}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityjEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/BroadcastGradientArguments:1*

Tidx0*
	keep_dims( *
T0
�
TEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/NegateNegSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Sum_1*
T0
�
WEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Reshape_1ReshapeTEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/NegateUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Shape_1*
T0*
_class
 *
Tshape0
�
YEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/GroupNoOpV^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/ReshapeX^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Reshape_1
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/ReshapeZ^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/Group*
T0*h
_class^
\Zloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Reshape
�
Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityWEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Reshape_1Z^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/Group*
T0*j
_class`
^\loc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Reshape_1
�
cEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/ReLU/PositivePartGradient/ReLUGradientReluGrad}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/WithControlDependencies/Identity/Identity.Estimator/Train/Model/ReLU_1/ReLU/PositivePart*
T0*
_class
 
|
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/ShapeConst*
valueB *
dtype0
�
UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Shape_1Shape.Estimator/Train/Model/ReLU_1/ReLU/NegativePart*
T0*
out_type0
�
hEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/BroadcastGradientArgumentsBroadcastGradientArgsSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/ShapeUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Shape_1*
T0
�
VEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/MultiplyMulEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/WithControlDependencies_1/Identity/Identity.Estimator/Train/Model/ReLU_1/ReLU/NegativePart*
T0
�
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/SumSumVEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/MultiplyhEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/BroadcastGradientArguments*

Tidx0*
	keep_dims( *
T0
�
UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/ReshapeReshapeQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/SumSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Shape*
T0*
_class
 *
Tshape0
�
XEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Multiply_1Mul%Estimator/Train/Model/ReLU_1/ConstantEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/WithControlDependencies_1/Identity/Identity*
T0
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Sum_1SumXEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Multiply_1jEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/BroadcastGradientArguments:1*

Tidx0*
	keep_dims( *
T0
�
WEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Reshape_1ReshapeSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Sum_1UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Shape_1*
T0*
_class
 *
Tshape0
�
YEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Tuple/GroupNoOpV^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/ReshapeX^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Reshape_1
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/ReshapeZ^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Tuple/Group*
T0*h
_class^
\Zloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Reshape
�
Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityWEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Reshape_1Z^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Tuple/Group*
T0*j
_class`
^\loc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Reshape_1
�
cEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/ReLU/NegativePartGradient/ReLUGradientReluGradEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Tuple/WithControlDependencies_1/Identity/Identity.Estimator/Train/Model/ReLU_1/ReLU/NegativePart*
T0
�
REstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/NegateGradient/NegateNegcEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/ReLU/NegativePartGradient/ReLUGradient*
T0
�
$Estimator/Train/Model/Gradients/AddNAddNcEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/ReLU/PositivePartGradient/ReLUGradientREstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/NegateGradient/Negate*
T0*v
_classl
jhloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/ReLU/PositivePartGradient/ReLUGradient*
N
t
+Estimator/Train/Model/Gradients/AddN_1/AddNIdentity$Estimator/Train/Model/Gradients/AddN*
T0*
_class
 
�
^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/BiasAddGradientBiasAddGrad+Estimator/Train/Model/Gradients/AddN_1/AddN*
_class
 *
data_formatNHWC*
T0
�
ZEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/GroupNoOp,^Estimator/Train/Model/Gradients/AddN_1/AddN_^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/BiasAddGradient
�
~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/WithControlDependencies/Identity/IdentityIdentity+Estimator/Train/Model/Gradients/AddN_1/AddN[^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/Group*
T0*>
_class4
20loc:@Estimator/Train/Model/Gradients/AddN_1/AddN
�
�Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentity^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/BiasAddGradient[^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/Group*q
_classg
ecloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/BiasAddGradient*
T0
�
TEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMulMatMul~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity$Estimator/Train/Model/ReadVariable_2*
T0*
_class
 *
transpose_a( *
transpose_b(
�
VEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMul_1MatMul#Estimator/Train/Model/ReLU/Subtract~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity*
transpose_a(*
transpose_b( *
T0*
_class
 
�
YEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/GroupNoOpU^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMulW^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMul_1
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityTEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMulZ^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/Group*
T0*g
_class]
[Yloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMul
�
Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityVEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMul_1Z^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/Group*
T0*i
_class_
][loc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMul_1
�
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/ShapeShape,Estimator/Train/Model/ReLU/ReLU/PositivePart*
T0*
out_type0
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Shape_1Shape#Estimator/Train/Model/ReLU/Multiply*
T0*
out_type0
�
fEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/BroadcastGradientArgumentsBroadcastGradientArgsQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/ShapeSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Shape_1*
T0
�
OEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/SumSum}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityfEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/BroadcastGradientArguments*

Tidx0*
	keep_dims( *
T0
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/ReshapeReshapeOEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/SumQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Shape*
T0*
_class
 *
Tshape0
�
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Sum_1Sum}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityhEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/BroadcastGradientArguments:1*

Tidx0*
	keep_dims( *
T0
�
REstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/NegateNegQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Sum_1*
T0
�
UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Reshape_1ReshapeREstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/NegateSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Shape_1*
T0*
_class
 *
Tshape0
�
WEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/GroupNoOpT^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/ReshapeV^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Reshape_1
�
{Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/WithControlDependencies/Identity/IdentityIdentitySEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/ReshapeX^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/Group*
T0*f
_class\
ZXloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Reshape
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Reshape_1X^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/Group*
T0*h
_class^
\Zloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Reshape_1
�
aEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/ReLU/PositivePartGradient/ReLUGradientReluGrad{Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/WithControlDependencies/Identity/Identity,Estimator/Train/Model/ReLU/ReLU/PositivePart*
T0*
_class
 
z
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/ShapeConst*
valueB *
dtype0
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Shape_1Shape,Estimator/Train/Model/ReLU/ReLU/NegativePart*
T0*
out_type0
�
fEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/BroadcastGradientArgumentsBroadcastGradientArgsQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/ShapeSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Shape_1*
T0
�
TEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/MultiplyMul}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/WithControlDependencies_1/Identity/Identity,Estimator/Train/Model/ReLU/ReLU/NegativePart*
T0
�
OEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/SumSumTEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/MultiplyfEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/BroadcastGradientArguments*
T0*

Tidx0*
	keep_dims( 
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/ReshapeReshapeOEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/SumQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Shape*
T0*
_class
 *
Tshape0
�
VEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Multiply_1Mul#Estimator/Train/Model/ReLU/Constant}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/WithControlDependencies_1/Identity/Identity*
T0
�
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Sum_1SumVEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Multiply_1hEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/BroadcastGradientArguments:1*

Tidx0*
	keep_dims( *
T0
�
UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Reshape_1ReshapeQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Sum_1SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Shape_1*
T0*
_class
 *
Tshape0
�
WEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Tuple/GroupNoOpT^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/ReshapeV^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Reshape_1
�
{Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Tuple/WithControlDependencies/Identity/IdentityIdentitySEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/ReshapeX^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Tuple/Group*f
_class\
ZXloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Reshape*
T0
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Reshape_1X^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Tuple/Group*h
_class^
\Zloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Reshape_1*
T0
�
aEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/ReLU/NegativePartGradient/ReLUGradientReluGrad}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Tuple/WithControlDependencies_1/Identity/Identity,Estimator/Train/Model/ReLU/ReLU/NegativePart*
T0
�
PEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/NegateGradient/NegateNegaEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/ReLU/NegativePartGradient/ReLUGradient*
T0
�
&Estimator/Train/Model/Gradients/AddN_2AddNaEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/ReLU/PositivePartGradient/ReLUGradientPEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/NegateGradient/Negate*t
_classj
hfloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/ReLU/PositivePartGradient/ReLUGradient*
N*
T0
v
+Estimator/Train/Model/Gradients/AddN_3/AddNIdentity&Estimator/Train/Model/Gradients/AddN_2*
T0*
_class
 
�
\Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/BiasAddGradientBiasAddGrad+Estimator/Train/Model/Gradients/AddN_3/AddN*
T0*
_class
 *
data_formatNHWC
�
XEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/GroupNoOp,^Estimator/Train/Model/Gradients/AddN_3/AddN]^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/BiasAddGradient
�
|Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/WithControlDependencies/Identity/IdentityIdentity+Estimator/Train/Model/Gradients/AddN_3/AddNY^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/Group*
T0*>
_class4
20loc:@Estimator/Train/Model/Gradients/AddN_3/AddN
�
~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentity\Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/BiasAddGradientY^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/Group*
T0*o
_classe
caloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/BiasAddGradient
�
REstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMulMatMul|Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity"Estimator/Train/Model/ReadVariable*
_class
 *
transpose_a( *
transpose_b(*
T0
�
TEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMul_1MatMul#Estimator/Train/Model/Input/Flatten|Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity*
_class
 *
transpose_a(*
transpose_b( *
T0
�
WEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/Tuple/GroupNoOpS^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMulU^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMul_1
�
{Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityREstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMulX^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/Tuple/Group*
T0*e
_class[
YWloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMul
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityTEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMul_1X^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/Tuple/Group*
T0*g
_class]
[Yloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMul_1
_
2Estimator/Train/Model/GradientDescent/LearningRateConst*
valueB
 "
�#<*
dtype0
�
�Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/GradientDescent/ApplyDenseResourceApplyGradientDescent@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights2Estimator/Train/Model/GradientDescent/LearningRateEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/WithControlDependencies_1/Identity/Identity*
use_locking( *
T0*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights
�
mEstimator/Train/Model/GradientDescent/Update/Input/Flatten/OutputLayer/Linear/Bias/GradientDescent/ApplyDenseResourceApplyGradientDescent%Input/Flatten/OutputLayer/Linear/Bias2Estimator/Train/Model/GradientDescent/LearningRate�Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/Identity*
use_locking( *
T0*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias
�
�Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/GradientDescent/ApplyDenseResourceApplyGradientDescentYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias2Estimator/Train/Model/GradientDescent/LearningRate~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/Identity*
use_locking( *
T0*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias
�
�Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/GradientDescent/ApplyDenseResourceApplyGradientDescent\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights2Estimator/Train/Model/GradientDescent/LearningRate}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/Tuple/WithControlDependencies_1/Identity/Identity*
use_locking( *
T0*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights
�
�Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/GradientDescent/ApplyDenseResourceApplyGradientDescent=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias2Estimator/Train/Model/GradientDescent/LearningRate�Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/Identity*
use_locking( *
T0*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias
�
pEstimator/Train/Model/GradientDescent/Update/Input/Flatten/OutputLayer/Linear/Weights/GradientDescent/ApplyDenseResourceApplyGradientDescent(Input/Flatten/OutputLayer/Linear/Weights2Estimator/Train/Model/GradientDescent/LearningRateEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/WithControlDependencies_1/Identity/Identity*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
use_locking( *
T0
�
,Estimator/Train/Model/GradientDescent/FinishNoOp�^Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/GradientDescent/ApplyDense�^Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/GradientDescent/ApplyDense�^Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/GradientDescent/ApplyDense�^Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/GradientDescent/ApplyDensen^Estimator/Train/Model/GradientDescent/Update/Input/Flatten/OutputLayer/Linear/Bias/GradientDescent/ApplyDenseq^Estimator/Train/Model/GradientDescent/Update/Input/Flatten/OutputLayer/Linear/Weights/GradientDescent/ApplyDense
�
+Estimator/Train/Model/GradientDescent/ShapeConst-^Estimator/Train/Model/GradientDescent/Finish*
valueB *
_class
loc:@global_step*
dtype0
�
3Estimator/Train/Model/GradientDescent/Ones/ConstantConst-^Estimator/Train/Model/GradientDescent/Finish*
valueB	 "       *
_class
loc:@global_step*
dtype0	
�
/Estimator/Train/Model/GradientDescent/Ones/FillFill+Estimator/Train/Model/GradientDescent/Shape3Estimator/Train/Model/GradientDescent/Ones/Constant*

index_type0*
_class
loc:@global_step*
T0	
�
5Estimator/Train/Model/GradientDescent/GradientDescentAssignAddVariableOpglobal_step/Estimator/Train/Model/GradientDescent/Ones/Fill*
_class
loc:@global_step*
dtype0	
�
Estimator/Infer/Model/IteratorIterator*&
output_shapes
:���������*
	container *
output_types
2*
shared_name 
�
 Estimator/Infer/Model/Input/NextIteratorGetNextEstimator/Infer/Model/Iterator*&
output_shapes
:���������*
output_types
2
P
Estimator/Infer/Model/ShapeConst*
valueB"����   *
dtype0
�
#Estimator/Infer/Model/Input/FlattenReshape Estimator/Infer/Model/Input/NextEstimator/Infer/Model/Shape*
T0*
Tshape0
�
"Estimator/Infer/Model/ReadVariableReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
$Estimator/Infer/Model/ReadVariable_1ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias
�
#Estimator/Infer/Model/Linear/MatMulMatMul#Estimator/Infer/Model/Input/Flatten"Estimator/Infer/Model/ReadVariable*
T0*
transpose_a( *
transpose_b( 
�
$Estimator/Infer/Model/Linear/AddBiasBiasAdd#Estimator/Infer/Model/Linear/MatMul$Estimator/Infer/Model/ReadVariable_1*
T0*
data_formatNHWC
c
,Estimator/Infer/Model/ReLU/ReLU/PositivePartRelu$Estimator/Infer/Model/Linear/AddBias*
T0
W
!Estimator/Infer/Model/ReLU/NegateNeg$Estimator/Infer/Model/Linear/AddBias*
T0
`
,Estimator/Infer/Model/ReLU/ReLU/NegativePartRelu!Estimator/Infer/Model/ReLU/Negate*
T0
P
#Estimator/Infer/Model/ReLU/ConstantConst*
valueB
 "���=*
dtype0
�
#Estimator/Infer/Model/ReLU/MultiplyMul#Estimator/Infer/Model/ReLU/Constant,Estimator/Infer/Model/ReLU/ReLU/NegativePart*
T0
�
#Estimator/Infer/Model/ReLU/SubtractSub,Estimator/Infer/Model/ReLU/ReLU/PositivePart#Estimator/Infer/Model/ReLU/Multiply*
T0
�
$Estimator/Infer/Model/ReadVariable_2ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
$Estimator/Infer/Model/ReadVariable_3ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
%Estimator/Infer/Model/Linear_1/MatMulMatMul#Estimator/Infer/Model/ReLU/Subtract$Estimator/Infer/Model/ReadVariable_2*
transpose_b( *
T0*
transpose_a( 
�
&Estimator/Infer/Model/Linear_1/AddBiasBiasAdd%Estimator/Infer/Model/Linear_1/MatMul$Estimator/Infer/Model/ReadVariable_3*
data_formatNHWC*
T0
g
.Estimator/Infer/Model/ReLU_1/ReLU/PositivePartRelu&Estimator/Infer/Model/Linear_1/AddBias*
T0
[
#Estimator/Infer/Model/ReLU_1/NegateNeg&Estimator/Infer/Model/Linear_1/AddBias*
T0
d
.Estimator/Infer/Model/ReLU_1/ReLU/NegativePartRelu#Estimator/Infer/Model/ReLU_1/Negate*
T0
R
%Estimator/Infer/Model/ReLU_1/ConstantConst*
valueB
 "���=*
dtype0
�
%Estimator/Infer/Model/ReLU_1/MultiplyMul%Estimator/Infer/Model/ReLU_1/Constant.Estimator/Infer/Model/ReLU_1/ReLU/NegativePart*
T0
�
%Estimator/Infer/Model/ReLU_1/SubtractSub.Estimator/Infer/Model/ReLU_1/ReLU/PositivePart%Estimator/Infer/Model/ReLU_1/Multiply*
T0
�
$Estimator/Infer/Model/ReadVariable_4ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
$Estimator/Infer/Model/ReadVariable_5ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
%Estimator/Infer/Model/Linear_2/MatMulMatMul%Estimator/Infer/Model/ReLU_1/Subtract$Estimator/Infer/Model/ReadVariable_4*
T0*
transpose_a( *
transpose_b( 
�
&Estimator/Infer/Model/Linear_2/AddBiasBiasAdd%Estimator/Infer/Model/Linear_2/MatMul$Estimator/Infer/Model/ReadVariable_5*
T0*
data_formatNHWC
�
!Estimator/Evaluate/Model/IteratorIterator*
output_types
2*
shared_name *5
output_shapes$
":���������:���������*
	container 
�
-Estimator/Evaluate/Model/Input_Input/Zip/NextIteratorGetNext!Estimator/Evaluate/Model/Iterator*
output_types
2*5
output_shapes$
":���������:���������
S
Estimator/Evaluate/Model/ShapeConst*
dtype0*
valueB"����   
�
&Estimator/Evaluate/Model/Input/FlattenReshape-Estimator/Evaluate/Model/Input_Input/Zip/NextEstimator/Evaluate/Model/Shape*
T0*
Tshape0
�
%Estimator/Evaluate/Model/ReadVariableReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
'Estimator/Evaluate/Model/ReadVariable_1ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
&Estimator/Evaluate/Model/Linear/MatMulMatMul&Estimator/Evaluate/Model/Input/Flatten%Estimator/Evaluate/Model/ReadVariable*
T0*
transpose_a( *
transpose_b( 
�
'Estimator/Evaluate/Model/Linear/AddBiasBiasAdd&Estimator/Evaluate/Model/Linear/MatMul'Estimator/Evaluate/Model/ReadVariable_1*
data_formatNHWC*
T0
i
/Estimator/Evaluate/Model/ReLU/ReLU/PositivePartRelu'Estimator/Evaluate/Model/Linear/AddBias*
T0
]
$Estimator/Evaluate/Model/ReLU/NegateNeg'Estimator/Evaluate/Model/Linear/AddBias*
T0
f
/Estimator/Evaluate/Model/ReLU/ReLU/NegativePartRelu$Estimator/Evaluate/Model/ReLU/Negate*
T0
S
&Estimator/Evaluate/Model/ReLU/ConstantConst*
valueB
 "���=*
dtype0
�
&Estimator/Evaluate/Model/ReLU/MultiplyMul&Estimator/Evaluate/Model/ReLU/Constant/Estimator/Evaluate/Model/ReLU/ReLU/NegativePart*
T0
�
&Estimator/Evaluate/Model/ReLU/SubtractSub/Estimator/Evaluate/Model/ReLU/ReLU/PositivePart&Estimator/Evaluate/Model/ReLU/Multiply*
T0
�
'Estimator/Evaluate/Model/ReadVariable_2ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
'Estimator/Evaluate/Model/ReadVariable_3ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
(Estimator/Evaluate/Model/Linear_1/MatMulMatMul&Estimator/Evaluate/Model/ReLU/Subtract'Estimator/Evaluate/Model/ReadVariable_2*
transpose_a( *
transpose_b( *
T0
�
)Estimator/Evaluate/Model/Linear_1/AddBiasBiasAdd(Estimator/Evaluate/Model/Linear_1/MatMul'Estimator/Evaluate/Model/ReadVariable_3*
T0*
data_formatNHWC
m
1Estimator/Evaluate/Model/ReLU_1/ReLU/PositivePartRelu)Estimator/Evaluate/Model/Linear_1/AddBias*
T0
a
&Estimator/Evaluate/Model/ReLU_1/NegateNeg)Estimator/Evaluate/Model/Linear_1/AddBias*
T0
j
1Estimator/Evaluate/Model/ReLU_1/ReLU/NegativePartRelu&Estimator/Evaluate/Model/ReLU_1/Negate*
T0
U
(Estimator/Evaluate/Model/ReLU_1/ConstantConst*
valueB
 "���=*
dtype0
�
(Estimator/Evaluate/Model/ReLU_1/MultiplyMul(Estimator/Evaluate/Model/ReLU_1/Constant1Estimator/Evaluate/Model/ReLU_1/ReLU/NegativePart*
T0
�
(Estimator/Evaluate/Model/ReLU_1/SubtractSub1Estimator/Evaluate/Model/ReLU_1/ReLU/PositivePart(Estimator/Evaluate/Model/ReLU_1/Multiply*
T0
�
'Estimator/Evaluate/Model/ReadVariable_4ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*
dtype0*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights
�
'Estimator/Evaluate/Model/ReadVariable_5ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
(Estimator/Evaluate/Model/Linear_2/MatMulMatMul(Estimator/Evaluate/Model/ReLU_1/Subtract'Estimator/Evaluate/Model/ReadVariable_4*
transpose_a( *
transpose_b( *
T0
�
)Estimator/Evaluate/Model/Linear_2/AddBiasBiasAdd(Estimator/Evaluate/Model/Linear_2/MatMul'Estimator/Evaluate/Model/ReadVariable_5*
T0*
data_formatNHWC
m
	eval_stepVarHandleOp*
shared_name	eval_step*
_class
 *
dtype0	*
	container *
shape: 
�
Beval_step/Initializer/eval_step/Initializer/ZerosInitializer/ShapeConst*
valueB *
_class
loc:@eval_step*
dtype0
�
Keval_step/Initializer/eval_step/Initializer/ZerosInitializer/Zeros/ConstantConst*
valueB	 "        *
_class
loc:@eval_step*
dtype0	
�
Geval_step/Initializer/eval_step/Initializer/ZerosInitializer/Zeros/FillFillBeval_step/Initializer/eval_step/Initializer/ZerosInitializer/ShapeKeval_step/Initializer/eval_step/Initializer/ZerosInitializer/Zeros/Constant*
T0	*

index_type0*
_class
loc:@eval_step
�
eval_step/InitializationAssignAssignVariableOp	eval_stepGeval_step/Initializer/eval_step/Initializer/ZerosInitializer/Zeros/Fill*
dtype0	
c
eval_step/Read/ReadVariableReadVariableOp	eval_step*
_class
loc:@eval_step*
dtype0	
8
eval_step/IsInitializedVarIsInitializedOp	eval_step
h
 eval_step/eval_step/ReadVariableReadVariableOp	eval_step*
_class
loc:@eval_step*
dtype0	
L
Estimator/Evaluate/ConstantConst*
valueB	 "       *
dtype0	
h
Estimator/Evaluate/AssignAddAssignAddVariableOp	eval_stepEstimator/Evaluate/Constant*
dtype0	
g
Estimator/Evaluate/AssignAdd_1ReadVariableOp	eval_step^Estimator/Evaluate/AssignAdd*
dtype0	
A
Estimator/Evaluate/GroupNoOp^Estimator/Evaluate/AssignAdd_1
b
Estimator/ReadVariableReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
J
Saver/ConstantConst*
_class
 *
valueB Bmodel*
dtype0
�
Saver/ReadVariableReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/ReadVariable_1ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/ReadVariable_2ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/ReadVariable_3ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/ReadVariable_4ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
Saver/ReadVariable_5ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
Saver/ReadVariable_6ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/ReadVariable_7ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/ReadVariable_8ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
Saver/ReadVariable_9ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
a
Saver/ReadVariable_10ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
a
Saver/ReadVariable_11ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
�
Saver/ReadVariable_12ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
Saver/ReadVariable_13ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
c
Saver/ReadVariable_14ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
c
Saver/ReadVariable_15ReadVariableOpglobal_epoch*
dtype0	*
_class
loc:@global_epoch
h
Saver/Constant_1Const*
dtype0*@
value7B5 B/_temp_fa9f4401-8f8c-4b4d-9131-e82ab6e86d17/part
Z
Saver/StringJoin
StringJoinSaver/ConstantSaver/Constant_1*
N*
	separator 
L
Saver/Constant_2Const"/device:CPU:0*
valueB
 "    *
dtype0
L
Saver/Constant_3Const"/device:CPU:0*
valueB
 "   *
dtype0
{
Saver/ShardedFilenameShardedFilenameSaver/StringJoinSaver/Constant_2Saver/Constant_3"/device:CPU:0*
_class
 
�
Saver/ReadVariable_16ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/ReadVariable_17ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
a
Saver/ReadVariable_18ReadVariableOpglobal_step*
dtype0	*
_class
loc:@global_step
�
Saver/ReadVariable_19ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
Saver/ReadVariable_20ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
Saver/ReadVariable_21ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/ReadVariable_22ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights
c
Saver/ReadVariable_23ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
�
Saver/Constant_4Const"/device:CPU:0*�
value�B�Bglobal_stepB%Input/Flatten/OutputLayer/Linear/BiasB=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/BiasB@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/WeightsBglobal_epochB(Input/Flatten/OutputLayer/Linear/WeightsBYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/BiasB\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
Z
Saver/Constant_5Const"/device:CPU:0*#
valueBB B B B B B B B *
dtype0
�

Saver/SaveSaveV2Saver/ShardedFilenameSaver/Constant_4Saver/Constant_5Saver/ReadVariable_18Saver/ReadVariable_20Saver/ReadVariable_16Saver/ReadVariable_22Saver/ReadVariable_23Saver/ReadVariable_19Saver/ReadVariable_17Saver/ReadVariable_21"/device:CPU:0*
dtypes

2		
�
/Saver/WithControlDependencies/Identity/IdentityIdentitySaver/ShardedFilename^Saver/Save"/device:CPU:0*
T0*(
_class
loc:@Saver/ShardedFilename
}
Saver/ShapeConst0^Saver/WithControlDependencies/Identity/Identity"/device:CPU:0*
valueB"   *
dtype0
b
Saver/ReshapeReshapeSaver/ShardedFilenameSaver/Shape"/device:CPU:0*
T0*
Tshape0
s
Saver/MergeV2CheckpointsMergeV2CheckpointsSaver/ReshapeSaver/Constant"/device:CPU:0*
delete_old_dirs(
�
1Saver/WithControlDependencies_1/Identity/IdentityIdentitySaver/Constant^Saver/MergeV2Checkpoints0^Saver/WithControlDependencies/Identity/Identity*
T0*!
_class
loc:@Saver/Constant
�
Saver/ReadVariable_24ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/ReadVariable_25ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/ReadVariable_26ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights
�
Saver/Constant_6Const"/device:CPU:0*q
valuehBfB\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
L
Saver/Constant_7Const"/device:CPU:0*
valueB
B *
dtype0
n
Saver/Restore	RestoreV2Saver/ConstantSaver/Constant_6Saver/Constant_7"/device:CPU:0*
dtypes
2
J
Saver/Identity/IdentityIdentitySaver/Restore"/device:CPU:0*
T0
�
Saver/AssignVariableAssignVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/WeightsSaver/Identity/Identity"/device:CPU:0*
dtype0
�
Saver/ReadVariable_27ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/ReadVariable_28ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/ReadVariable_29ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/Constant_8Const"/device:CPU:0*R
valueIBGB=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
L
Saver/Constant_9Const"/device:CPU:0*
valueB
B *
dtype0
p
Saver/Restore_1	RestoreV2Saver/ConstantSaver/Constant_8Saver/Constant_9"/device:CPU:0*
dtypes
2
N
Saver/Identity_1/IdentityIdentitySaver/Restore_1"/device:CPU:0*
T0
�
Saver/AssignVariable_1AssignVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/BiasSaver/Identity_1/Identity"/device:CPU:0*
dtype0
�
Saver/ReadVariable_30ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
Saver/ReadVariable_31ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
Saver/ReadVariable_32ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
r
Saver/Constant_10Const"/device:CPU:0*:
value1B/B%Input/Flatten/OutputLayer/Linear/Bias*
dtype0
M
Saver/Constant_11Const"/device:CPU:0*
dtype0*
valueB
B 
r
Saver/Restore_2	RestoreV2Saver/ConstantSaver/Constant_10Saver/Constant_11"/device:CPU:0*
dtypes
2
N
Saver/Identity_2/IdentityIdentitySaver/Restore_2"/device:CPU:0*
T0
�
Saver/AssignVariable_2AssignVariableOp%Input/Flatten/OutputLayer/Linear/BiasSaver/Identity_2/Identity"/device:CPU:0*
dtype0
�
Saver/ReadVariable_33ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/ReadVariable_34ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/ReadVariable_35ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/Constant_12Const"/device:CPU:0*U
valueLBJB@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
M
Saver/Constant_13Const"/device:CPU:0*
valueB
B *
dtype0
r
Saver/Restore_3	RestoreV2Saver/ConstantSaver/Constant_12Saver/Constant_13"/device:CPU:0*
dtypes
2
N
Saver/Identity_3/IdentityIdentitySaver/Restore_3"/device:CPU:0*
T0
�
Saver/AssignVariable_3AssignVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/WeightsSaver/Identity_3/Identity"/device:CPU:0*
dtype0
�
Saver/ReadVariable_36ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
Saver/ReadVariable_37ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
Saver/ReadVariable_38ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
Saver/Constant_14Const"/device:CPU:0*n
valueeBcBYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
M
Saver/Constant_15Const"/device:CPU:0*
valueB
B *
dtype0
r
Saver/Restore_4	RestoreV2Saver/ConstantSaver/Constant_14Saver/Constant_15"/device:CPU:0*
dtypes
2
N
Saver/Identity_4/IdentityIdentitySaver/Restore_4"/device:CPU:0*
T0
�
Saver/AssignVariable_4AssignVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/BiasSaver/Identity_4/Identity"/device:CPU:0*
dtype0
a
Saver/ReadVariable_39ReadVariableOpglobal_step*
dtype0	*
_class
loc:@global_step
a
Saver/ReadVariable_40ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
a
Saver/ReadVariable_41ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
X
Saver/Constant_16Const"/device:CPU:0* 
valueBBglobal_step*
dtype0
M
Saver/Constant_17Const"/device:CPU:0*
valueB
B *
dtype0
r
Saver/Restore_5	RestoreV2Saver/ConstantSaver/Constant_16Saver/Constant_17"/device:CPU:0*
dtypes
2	
N
Saver/Identity_5/IdentityIdentitySaver/Restore_5"/device:CPU:0*
T0	
n
Saver/AssignVariable_5AssignVariableOpglobal_stepSaver/Identity_5/Identity"/device:CPU:0*
dtype0	
�
Saver/ReadVariable_42ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
Saver/ReadVariable_43ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
Saver/ReadVariable_44ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
u
Saver/Constant_18Const"/device:CPU:0*=
value4B2B(Input/Flatten/OutputLayer/Linear/Weights*
dtype0
M
Saver/Constant_19Const"/device:CPU:0*
valueB
B *
dtype0
r
Saver/Restore_6	RestoreV2Saver/ConstantSaver/Constant_18Saver/Constant_19"/device:CPU:0*
dtypes
2
N
Saver/Identity_6/IdentityIdentitySaver/Restore_6"/device:CPU:0*
T0
�
Saver/AssignVariable_6AssignVariableOp(Input/Flatten/OutputLayer/Linear/WeightsSaver/Identity_6/Identity"/device:CPU:0*
dtype0
c
Saver/ReadVariable_45ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
c
Saver/ReadVariable_46ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
c
Saver/ReadVariable_47ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
Y
Saver/Constant_20Const"/device:CPU:0*!
valueBBglobal_epoch*
dtype0
M
Saver/Constant_21Const"/device:CPU:0*
dtype0*
valueB
B 
r
Saver/Restore_7	RestoreV2Saver/ConstantSaver/Constant_20Saver/Constant_21"/device:CPU:0*
dtypes
2	
N
Saver/Identity_7/IdentityIdentitySaver/Restore_7"/device:CPU:0*
T0	
o
Saver/AssignVariable_7AssignVariableOpglobal_epochSaver/Identity_7/Identity"/device:CPU:0*
dtype0	
�
Saver/GroupNoOp^Saver/AssignVariable^Saver/AssignVariable_1^Saver/AssignVariable_2^Saver/AssignVariable_3^Saver/AssignVariable_4^Saver/AssignVariable_5^Saver/AssignVariable_6^Saver/AssignVariable_7"/device:CPU:0
2
Saver/Group_1NoOp^Saver/Group"/device:CPU:0
X
ReadVariableReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	 "����     ����	/�t��A�A"ٟ

s
global_epochVarHandleOp*
shape: *
shared_nameglobal_epoch*
_class
 *
dtype0	*
	container 
�
Hglobal_epoch/Initializer/global_epoch/Initializer/ZerosInitializer/ShapeConst*
valueB *
_class
loc:@global_epoch*
dtype0
�
Qglobal_epoch/Initializer/global_epoch/Initializer/ZerosInitializer/Zeros/ConstantConst*
valueB	 "        *
_class
loc:@global_epoch*
dtype0	
�
Mglobal_epoch/Initializer/global_epoch/Initializer/ZerosInitializer/Zeros/FillFillHglobal_epoch/Initializer/global_epoch/Initializer/ZerosInitializer/ShapeQglobal_epoch/Initializer/global_epoch/Initializer/ZerosInitializer/Zeros/Constant*
T0	*

index_type0*
_class
loc:@global_epoch
�
!global_epoch/InitializationAssignAssignVariableOpglobal_epochMglobal_epoch/Initializer/global_epoch/Initializer/ZerosInitializer/Zeros/Fill*
dtype0	
l
global_epoch/Read/ReadVariableReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
>
global_epoch/IsInitializedVarIsInitializedOpglobal_epoch
t
&global_epoch/global_epoch/ReadVariableReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
q
global_stepVarHandleOp*
shape: *
shared_nameglobal_step*
_class
 *
dtype0	*
	container 
�
Fglobal_step/Initializer/global_step/Initializer/ZerosInitializer/ShapeConst*
valueB *
_class
loc:@global_step*
dtype0
�
Oglobal_step/Initializer/global_step/Initializer/ZerosInitializer/Zeros/ConstantConst*
valueB	 "        *
_class
loc:@global_step*
dtype0	
�
Kglobal_step/Initializer/global_step/Initializer/ZerosInitializer/Zeros/FillFillFglobal_step/Initializer/global_step/Initializer/ZerosInitializer/ShapeOglobal_step/Initializer/global_step/Initializer/ZerosInitializer/Zeros/Constant*
T0	*

index_type0*
_class
loc:@global_step
�
 global_step/InitializationAssignAssignVariableOpglobal_stepKglobal_step/Initializer/global_step/Initializer/ZerosInitializer/Zeros/Fill*
dtype0	
i
global_step/Read/ReadVariableReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
<
global_step/IsInitializedVarIsInitializedOpglobal_step
p
$global_step/global_step/ReadVariableReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
�
Estimator/Train/Model/IteratorIterator*
_class
 *
	container *
output_types
2*
shared_name *5
output_shapes$
":���������:���������
�
*Estimator/Train/Model/Input_Input/Zip/NextIteratorGetNextEstimator/Train/Model/Iterator*
output_types
2*5
output_shapes$
":���������:���������
P
Estimator/Train/Model/ShapeConst*
valueB"����   *
dtype0
�
#Estimator/Train/Model/Input/FlattenReshape*Estimator/Train/Model/Input_Input/Zip/NextEstimator/Train/Model/Shape*
Tshape0*
T0
�
\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/WeightsVarHandleOp*
_class
 *
dtype0*
	container *
shape
:@*m
shared_name^\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/ShapeConst*
valueB"   @   *o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/ConstantConst*
valueB
 "    *o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Constant_1Const*
valueB
 "  �?*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/RandomNormalInitializerRandomStandardNormal�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Shape*

seed *
T0*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0*
seed2 
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/MultiplyMul�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/RandomNormalInitializer�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Constant_1*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
T0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/AddAdd�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Multiply�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Constant*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
T0
�
qInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/InitializationAssignAssignVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Add*
dtype0
�
nInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Read/ReadVariableReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
jInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/IsInitializedVarIsInitializedOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/ReadVariableReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
"Estimator/Train/Model/ReadVariableReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
YInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/BiasVarHandleOp*
_class
 *
dtype0*
	container *
shape:@*j
shared_name[YInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/ShapeConst*
valueB"@   *l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/ConstantConst*
valueB
 "    *l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Constant_1Const*
valueB
 "  �?*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/RandomNormalInitializerRandomStandardNormal�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Shape*
T0*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0*
seed2 *

seed 
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/MultiplyMul�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/RandomNormalInitializer�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Constant_1*
T0*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/AddAdd�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Multiply�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Constant*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
T0
�
nInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/InitializationAssignAssignVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Add*
dtype0
�
kInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Read/ReadVariableReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
gInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/IsInitializedVarIsInitializedOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/ReadVariableReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
$Estimator/Train/Model/ReadVariable_1ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
#Estimator/Train/Model/Linear/MatMulMatMul#Estimator/Train/Model/Input/Flatten"Estimator/Train/Model/ReadVariable*
T0*
transpose_a( *
transpose_b( 
�
$Estimator/Train/Model/Linear/AddBiasBiasAdd#Estimator/Train/Model/Linear/MatMul$Estimator/Train/Model/ReadVariable_1*
T0*
data_formatNHWC
c
,Estimator/Train/Model/ReLU/ReLU/PositivePartRelu$Estimator/Train/Model/Linear/AddBias*
T0
W
!Estimator/Train/Model/ReLU/NegateNeg$Estimator/Train/Model/Linear/AddBias*
T0
`
,Estimator/Train/Model/ReLU/ReLU/NegativePartRelu!Estimator/Train/Model/ReLU/Negate*
T0
P
#Estimator/Train/Model/ReLU/ConstantConst*
valueB
 "���=*
dtype0
�
#Estimator/Train/Model/ReLU/MultiplyMul#Estimator/Train/Model/ReLU/Constant,Estimator/Train/Model/ReLU/ReLU/NegativePart*
T0
�
#Estimator/Train/Model/ReLU/SubtractSub,Estimator/Train/Model/ReLU/ReLU/PositivePart#Estimator/Train/Model/ReLU/Multiply*
T0
�
@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/WeightsVarHandleOp*
_class
 *
dtype0*
	container *
shape
:@@*Q
shared_nameB@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/ShapeConst*
valueB"@   @   *S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/ConstantConst*
valueB
 "    *S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Constant_1Const*
valueB
 "  �?*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/RandomNormalInitializerRandomStandardNormal�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Shape*

seed *
T0*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0*
seed2 
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/MultiplyMul�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/RandomNormalInitializer�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Constant_1*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
T0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/AddAdd�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Multiply�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Constant*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
T0
�
UInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/InitializationAssignAssignVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Add*
dtype0
�
RInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Read/ReadVariableReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
NInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/IsInitializedVarIsInitializedOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/ReadVariableReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
$Estimator/Train/Model/ReadVariable_2ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/BiasVarHandleOp*
_class
 *
dtype0*
	container *
shape:@*N
shared_name?=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/ShapeConst*
valueB"@   *P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/ConstantConst*
valueB
 "    *P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Constant_1Const*
valueB
 "  �?*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/RandomNormalInitializerRandomStandardNormal�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Shape*
T0*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0*
seed2 *

seed 
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/MultiplyMul�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/RandomNormalInitializer�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Constant_1*
T0*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/AddAdd�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Multiply�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Constant*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
T0
�
RInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/InitializationAssignAssignVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Add*
dtype0
�
OInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Read/ReadVariableReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
KInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/IsInitializedVarIsInitializedOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/ReadVariableReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
$Estimator/Train/Model/ReadVariable_3ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
%Estimator/Train/Model/Linear_1/MatMulMatMul#Estimator/Train/Model/ReLU/Subtract$Estimator/Train/Model/ReadVariable_2*
T0*
transpose_a( *
transpose_b( 
�
&Estimator/Train/Model/Linear_1/AddBiasBiasAdd%Estimator/Train/Model/Linear_1/MatMul$Estimator/Train/Model/ReadVariable_3*
T0*
data_formatNHWC
g
.Estimator/Train/Model/ReLU_1/ReLU/PositivePartRelu&Estimator/Train/Model/Linear_1/AddBias*
T0
[
#Estimator/Train/Model/ReLU_1/NegateNeg&Estimator/Train/Model/Linear_1/AddBias*
T0
d
.Estimator/Train/Model/ReLU_1/ReLU/NegativePartRelu#Estimator/Train/Model/ReLU_1/Negate*
T0
R
%Estimator/Train/Model/ReLU_1/ConstantConst*
valueB
 "���=*
dtype0
�
%Estimator/Train/Model/ReLU_1/MultiplyMul%Estimator/Train/Model/ReLU_1/Constant.Estimator/Train/Model/ReLU_1/ReLU/NegativePart*
T0
�
%Estimator/Train/Model/ReLU_1/SubtractSub.Estimator/Train/Model/ReLU_1/ReLU/PositivePart%Estimator/Train/Model/ReLU_1/Multiply*
T0
�
(Input/Flatten/OutputLayer/Linear/WeightsVarHandleOp*
shape
:@*9
shared_name*(Input/Flatten/OutputLayer/Linear/Weights*
_class
 *
dtype0*
	container 
�
oInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/ShapeConst*
valueB"@      *;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
rInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/ConstantConst*
valueB
 "    *;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
tInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/Constant_1Const*
valueB
 "  �?*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
�Input/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/RandomNormalInitializerRandomStandardNormaloInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/Shape*
seed2 *

seed *
T0*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
rInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/MultiplyMul�Input/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/RandomNormalInitializertInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/Constant_1*
T0*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights
�
mInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/AddAddrInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/MultiplyrInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/Constant*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
T0
�
=Input/Flatten/OutputLayer/Linear/Weights/InitializationAssignAssignVariableOp(Input/Flatten/OutputLayer/Linear/WeightsmInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/Add*
dtype0
�
:Input/Flatten/OutputLayer/Linear/Weights/Read/ReadVariableReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
v
6Input/Flatten/OutputLayer/Linear/Weights/IsInitializedVarIsInitializedOp(Input/Flatten/OutputLayer/Linear/Weights
�
^Input/Flatten/OutputLayer/Linear/Weights/Input/Flatten/OutputLayer/Linear/Weights/ReadVariableReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
$Estimator/Train/Model/ReadVariable_4ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
%Input/Flatten/OutputLayer/Linear/BiasVarHandleOp*
shape:*6
shared_name'%Input/Flatten/OutputLayer/Linear/Bias*
_class
 *
dtype0*
	container 
�
iInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/ShapeConst*
valueB"   *8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
lInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/ConstantConst*
valueB
 "    *8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
nInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/Constant_1Const*
valueB
 "  �?*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
{Input/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/RandomNormalInitializerRandomStandardNormaliInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/Shape*
seed2 *

seed *
T0*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
lInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/MultiplyMul{Input/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/RandomNormalInitializernInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/Constant_1*
T0*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias
�
gInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/AddAddlInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/MultiplylInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/Constant*
T0*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias
�
:Input/Flatten/OutputLayer/Linear/Bias/InitializationAssignAssignVariableOp%Input/Flatten/OutputLayer/Linear/BiasgInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/Add*
dtype0
�
7Input/Flatten/OutputLayer/Linear/Bias/Read/ReadVariableReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
p
3Input/Flatten/OutputLayer/Linear/Bias/IsInitializedVarIsInitializedOp%Input/Flatten/OutputLayer/Linear/Bias
�
XInput/Flatten/OutputLayer/Linear/Bias/Input/Flatten/OutputLayer/Linear/Bias/ReadVariableReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
$Estimator/Train/Model/ReadVariable_5ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
%Estimator/Train/Model/Linear_2/MatMulMatMul%Estimator/Train/Model/ReLU_1/Subtract$Estimator/Train/Model/ReadVariable_4*
T0*
transpose_a( *
transpose_b( 
�
&Estimator/Train/Model/Linear_2/AddBiasBiasAdd%Estimator/Train/Model/Linear_2/MatMul$Estimator/Train/Model/ReadVariable_5*
data_formatNHWC*
T0
�
Estimator/Train/Model/SubtractSub&Estimator/Train/Model/Linear_2/AddBias,Estimator/Train/Model/Input_Input/Zip/Next:1*
T0
P
Estimator/Train/Model/Loss/L2L2LossEstimator/Train/Model/Subtract*
T0
Z
1Estimator/Train/Model/Gradients/Gradients_0/ShapeConst*
valueB *
dtype0
f
9Estimator/Train/Model/Gradients/Gradients_0/Ones/ConstantConst*
valueB
 "  �?*
dtype0
�
5Estimator/Train/Model/Gradients/Gradients_0/Ones/FillFill1Estimator/Train/Model/Gradients/Gradients_0/Shape9Estimator/Train/Model/Gradients/Gradients_0/Ones/Constant*
T0*

index_type0
�
NEstimator/Train/Model/Gradients/Estimator/Train/Model/Loss/L2Gradient/MultiplyMulEstimator/Train/Model/Subtract5Estimator/Train/Model/Gradients/Gradients_0/Ones/Fill*
T0
�
LEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/ShapeShape&Estimator/Train/Model/Linear_2/AddBias*
T0*
out_type0
�
NEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Shape_1Shape,Estimator/Train/Model/Input_Input/Zip/Next:1*
out_type0*
T0
�
aEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/BroadcastGradientArgumentsBroadcastGradientArgsLEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/ShapeNEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Shape_1*
T0
�
JEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/SumSumNEstimator/Train/Model/Gradients/Estimator/Train/Model/Loss/L2Gradient/MultiplyaEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/BroadcastGradientArguments*

Tidx0*
	keep_dims( *
T0
�
NEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/ReshapeReshapeJEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/SumLEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Shape*
T0*
Tshape0*
_class
 
�
LEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Sum_1SumNEstimator/Train/Model/Gradients/Estimator/Train/Model/Loss/L2Gradient/MultiplycEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/BroadcastGradientArguments:1*

Tidx0*
	keep_dims( *
T0
�
MEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/NegateNegLEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Sum_1*
T0
�
PEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Reshape_1ReshapeMEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/NegateNEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Shape_1*
T0*
_class
 *
Tshape0
�
REstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/GroupNoOpO^Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/ReshapeQ^Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Reshape_1
�
vEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityNEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/ReshapeS^Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/Group*
T0*a
_classW
USloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Reshape
�
xEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityPEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Reshape_1S^Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/Group*
T0*c
_classY
WUloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Reshape_1
�
^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/BiasAddGradientBiasAddGradvEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/WithControlDependencies/Identity/Identity*
_class
 *
data_formatNHWC*
T0
�
ZEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/GroupNoOp_^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/BiasAddGradientw^Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/WithControlDependencies/Identity/Identity
�
~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityvEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/WithControlDependencies/Identity/Identity[^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/Group*�
_class�
�Sloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Reshape{loc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/WithControlDependencies/Identity/Identity*
T0
�
�Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentity^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/BiasAddGradient[^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/Group*q
_classg
ecloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/BiasAddGradient*
T0
�
TEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMulMatMul~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity$Estimator/Train/Model/ReadVariable_4*
_class
 *
transpose_a( *
transpose_b(*
T0
�
VEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul_1MatMul%Estimator/Train/Model/ReLU_1/Subtract~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity*
T0*
_class
 *
transpose_a(*
transpose_b( 
�
YEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/GroupNoOpU^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMulW^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul_1
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityTEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMulZ^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/Group*
T0*g
_class]
[Yloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul
�
Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityVEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul_1Z^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/Group*
T0*i
_class_
][loc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul_1
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/ShapeShape.Estimator/Train/Model/ReLU_1/ReLU/PositivePart*
T0*
out_type0
�
UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Shape_1Shape%Estimator/Train/Model/ReLU_1/Multiply*
T0*
out_type0
�
hEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/BroadcastGradientArgumentsBroadcastGradientArgsSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/ShapeUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Shape_1*
T0
�
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/SumSum}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityhEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/BroadcastGradientArguments*
T0*

Tidx0*
	keep_dims( 
�
UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/ReshapeReshapeQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/SumSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Shape*
_class
 *
Tshape0*
T0
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Sum_1Sum}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityjEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/BroadcastGradientArguments:1*

Tidx0*
	keep_dims( *
T0
�
TEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/NegateNegSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Sum_1*
T0
�
WEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Reshape_1ReshapeTEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/NegateUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Shape_1*
T0*
_class
 *
Tshape0
�
YEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/GroupNoOpV^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/ReshapeX^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Reshape_1
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/ReshapeZ^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/Group*
T0*h
_class^
\Zloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Reshape
�
Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityWEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Reshape_1Z^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/Group*
T0*j
_class`
^\loc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Reshape_1
�
cEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/ReLU/PositivePartGradient/ReLUGradientReluGrad}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/WithControlDependencies/Identity/Identity.Estimator/Train/Model/ReLU_1/ReLU/PositivePart*
T0*
_class
 
|
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/ShapeConst*
valueB *
dtype0
�
UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Shape_1Shape.Estimator/Train/Model/ReLU_1/ReLU/NegativePart*
T0*
out_type0
�
hEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/BroadcastGradientArgumentsBroadcastGradientArgsSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/ShapeUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Shape_1*
T0
�
VEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/MultiplyMulEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/WithControlDependencies_1/Identity/Identity.Estimator/Train/Model/ReLU_1/ReLU/NegativePart*
T0
�
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/SumSumVEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/MultiplyhEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/BroadcastGradientArguments*

Tidx0*
	keep_dims( *
T0
�
UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/ReshapeReshapeQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/SumSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Shape*
_class
 *
Tshape0*
T0
�
XEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Multiply_1Mul%Estimator/Train/Model/ReLU_1/ConstantEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/WithControlDependencies_1/Identity/Identity*
T0
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Sum_1SumXEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Multiply_1jEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/BroadcastGradientArguments:1*

Tidx0*
	keep_dims( *
T0
�
WEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Reshape_1ReshapeSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Sum_1UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Shape_1*
_class
 *
Tshape0*
T0
�
YEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Tuple/GroupNoOpV^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/ReshapeX^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Reshape_1
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/ReshapeZ^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Tuple/Group*
T0*h
_class^
\Zloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Reshape
�
Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityWEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Reshape_1Z^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Tuple/Group*j
_class`
^\loc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Reshape_1*
T0
�
cEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/ReLU/NegativePartGradient/ReLUGradientReluGradEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Tuple/WithControlDependencies_1/Identity/Identity.Estimator/Train/Model/ReLU_1/ReLU/NegativePart*
T0
�
REstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/NegateGradient/NegateNegcEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/ReLU/NegativePartGradient/ReLUGradient*
T0
�
$Estimator/Train/Model/Gradients/AddNAddNcEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/ReLU/PositivePartGradient/ReLUGradientREstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/NegateGradient/Negate*
T0*v
_classl
jhloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/ReLU/PositivePartGradient/ReLUGradient*
N
t
+Estimator/Train/Model/Gradients/AddN_1/AddNIdentity$Estimator/Train/Model/Gradients/AddN*
_class
 *
T0
�
^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/BiasAddGradientBiasAddGrad+Estimator/Train/Model/Gradients/AddN_1/AddN*
T0*
_class
 *
data_formatNHWC
�
ZEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/GroupNoOp,^Estimator/Train/Model/Gradients/AddN_1/AddN_^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/BiasAddGradient
�
~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/WithControlDependencies/Identity/IdentityIdentity+Estimator/Train/Model/Gradients/AddN_1/AddN[^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/Group*
T0*>
_class4
20loc:@Estimator/Train/Model/Gradients/AddN_1/AddN
�
�Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentity^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/BiasAddGradient[^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/Group*q
_classg
ecloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/BiasAddGradient*
T0
�
TEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMulMatMul~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity$Estimator/Train/Model/ReadVariable_2*
transpose_a( *
transpose_b(*
T0*
_class
 
�
VEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMul_1MatMul#Estimator/Train/Model/ReLU/Subtract~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity*
T0*
_class
 *
transpose_a(*
transpose_b( 
�
YEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/GroupNoOpU^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMulW^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMul_1
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityTEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMulZ^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/Group*g
_class]
[Yloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMul*
T0
�
Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityVEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMul_1Z^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/Group*i
_class_
][loc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMul_1*
T0
�
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/ShapeShape,Estimator/Train/Model/ReLU/ReLU/PositivePart*
out_type0*
T0
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Shape_1Shape#Estimator/Train/Model/ReLU/Multiply*
T0*
out_type0
�
fEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/BroadcastGradientArgumentsBroadcastGradientArgsQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/ShapeSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Shape_1*
T0
�
OEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/SumSum}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityfEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/BroadcastGradientArguments*

Tidx0*
	keep_dims( *
T0
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/ReshapeReshapeOEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/SumQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Shape*
T0*
_class
 *
Tshape0
�
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Sum_1Sum}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityhEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/BroadcastGradientArguments:1*

Tidx0*
	keep_dims( *
T0
�
REstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/NegateNegQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Sum_1*
T0
�
UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Reshape_1ReshapeREstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/NegateSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Shape_1*
_class
 *
Tshape0*
T0
�
WEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/GroupNoOpT^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/ReshapeV^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Reshape_1
�
{Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/WithControlDependencies/Identity/IdentityIdentitySEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/ReshapeX^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/Group*
T0*f
_class\
ZXloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Reshape
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Reshape_1X^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/Group*
T0*h
_class^
\Zloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Reshape_1
�
aEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/ReLU/PositivePartGradient/ReLUGradientReluGrad{Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/WithControlDependencies/Identity/Identity,Estimator/Train/Model/ReLU/ReLU/PositivePart*
T0*
_class
 
z
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/ShapeConst*
valueB *
dtype0
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Shape_1Shape,Estimator/Train/Model/ReLU/ReLU/NegativePart*
T0*
out_type0
�
fEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/BroadcastGradientArgumentsBroadcastGradientArgsQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/ShapeSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Shape_1*
T0
�
TEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/MultiplyMul}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/WithControlDependencies_1/Identity/Identity,Estimator/Train/Model/ReLU/ReLU/NegativePart*
T0
�
OEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/SumSumTEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/MultiplyfEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/BroadcastGradientArguments*

Tidx0*
	keep_dims( *
T0
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/ReshapeReshapeOEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/SumQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Shape*
T0*
_class
 *
Tshape0
�
VEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Multiply_1Mul#Estimator/Train/Model/ReLU/Constant}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/WithControlDependencies_1/Identity/Identity*
T0
�
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Sum_1SumVEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Multiply_1hEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/BroadcastGradientArguments:1*

Tidx0*
	keep_dims( *
T0
�
UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Reshape_1ReshapeQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Sum_1SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Shape_1*
T0*
_class
 *
Tshape0
�
WEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Tuple/GroupNoOpT^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/ReshapeV^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Reshape_1
�
{Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Tuple/WithControlDependencies/Identity/IdentityIdentitySEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/ReshapeX^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Tuple/Group*
T0*f
_class\
ZXloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Reshape
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Reshape_1X^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Tuple/Group*
T0*h
_class^
\Zloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Reshape_1
�
aEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/ReLU/NegativePartGradient/ReLUGradientReluGrad}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Tuple/WithControlDependencies_1/Identity/Identity,Estimator/Train/Model/ReLU/ReLU/NegativePart*
T0
�
PEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/NegateGradient/NegateNegaEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/ReLU/NegativePartGradient/ReLUGradient*
T0
�
&Estimator/Train/Model/Gradients/AddN_2AddNaEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/ReLU/PositivePartGradient/ReLUGradientPEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/NegateGradient/Negate*t
_classj
hfloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/ReLU/PositivePartGradient/ReLUGradient*
N*
T0
v
+Estimator/Train/Model/Gradients/AddN_3/AddNIdentity&Estimator/Train/Model/Gradients/AddN_2*
T0*
_class
 
�
\Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/BiasAddGradientBiasAddGrad+Estimator/Train/Model/Gradients/AddN_3/AddN*
T0*
_class
 *
data_formatNHWC
�
XEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/GroupNoOp,^Estimator/Train/Model/Gradients/AddN_3/AddN]^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/BiasAddGradient
�
|Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/WithControlDependencies/Identity/IdentityIdentity+Estimator/Train/Model/Gradients/AddN_3/AddNY^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/Group*
T0*>
_class4
20loc:@Estimator/Train/Model/Gradients/AddN_3/AddN
�
~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentity\Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/BiasAddGradientY^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/Group*
T0*o
_classe
caloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/BiasAddGradient
�
REstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMulMatMul|Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity"Estimator/Train/Model/ReadVariable*
transpose_a( *
transpose_b(*
T0*
_class
 
�
TEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMul_1MatMul#Estimator/Train/Model/Input/Flatten|Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity*
transpose_a(*
transpose_b( *
T0*
_class
 
�
WEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/Tuple/GroupNoOpS^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMulU^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMul_1
�
{Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityREstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMulX^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/Tuple/Group*e
_class[
YWloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMul*
T0
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityTEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMul_1X^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/Tuple/Group*
T0*g
_class]
[Yloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMul_1
_
2Estimator/Train/Model/GradientDescent/LearningRateConst*
valueB
 "
�#<*
dtype0
�
�Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/GradientDescent/ApplyDenseResourceApplyGradientDescent@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights2Estimator/Train/Model/GradientDescent/LearningRateEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/WithControlDependencies_1/Identity/Identity*
use_locking( *
T0*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights
�
mEstimator/Train/Model/GradientDescent/Update/Input/Flatten/OutputLayer/Linear/Bias/GradientDescent/ApplyDenseResourceApplyGradientDescent%Input/Flatten/OutputLayer/Linear/Bias2Estimator/Train/Model/GradientDescent/LearningRate�Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/Identity*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
use_locking( *
T0
�
�Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/GradientDescent/ApplyDenseResourceApplyGradientDescentYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias2Estimator/Train/Model/GradientDescent/LearningRate~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/Identity*
use_locking( *
T0*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias
�
�Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/GradientDescent/ApplyDenseResourceApplyGradientDescent\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights2Estimator/Train/Model/GradientDescent/LearningRate}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/Tuple/WithControlDependencies_1/Identity/Identity*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
use_locking( *
T0
�
�Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/GradientDescent/ApplyDenseResourceApplyGradientDescent=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias2Estimator/Train/Model/GradientDescent/LearningRate�Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/Identity*
use_locking( *
T0*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias
�
pEstimator/Train/Model/GradientDescent/Update/Input/Flatten/OutputLayer/Linear/Weights/GradientDescent/ApplyDenseResourceApplyGradientDescent(Input/Flatten/OutputLayer/Linear/Weights2Estimator/Train/Model/GradientDescent/LearningRateEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/WithControlDependencies_1/Identity/Identity*
use_locking( *
T0*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights
�
,Estimator/Train/Model/GradientDescent/FinishNoOp�^Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/GradientDescent/ApplyDense�^Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/GradientDescent/ApplyDense�^Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/GradientDescent/ApplyDense�^Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/GradientDescent/ApplyDensen^Estimator/Train/Model/GradientDescent/Update/Input/Flatten/OutputLayer/Linear/Bias/GradientDescent/ApplyDenseq^Estimator/Train/Model/GradientDescent/Update/Input/Flatten/OutputLayer/Linear/Weights/GradientDescent/ApplyDense
�
+Estimator/Train/Model/GradientDescent/ShapeConst-^Estimator/Train/Model/GradientDescent/Finish*
valueB *
_class
loc:@global_step*
dtype0
�
3Estimator/Train/Model/GradientDescent/Ones/ConstantConst-^Estimator/Train/Model/GradientDescent/Finish*
valueB	 "       *
_class
loc:@global_step*
dtype0	
�
/Estimator/Train/Model/GradientDescent/Ones/FillFill+Estimator/Train/Model/GradientDescent/Shape3Estimator/Train/Model/GradientDescent/Ones/Constant*
T0	*

index_type0*
_class
loc:@global_step
�
5Estimator/Train/Model/GradientDescent/GradientDescentAssignAddVariableOpglobal_step/Estimator/Train/Model/GradientDescent/Ones/Fill*
_class
loc:@global_step*
dtype0	
�
Estimator/Infer/Model/IteratorIterator*
	container *
output_types
2*
shared_name *&
output_shapes
:���������
�
 Estimator/Infer/Model/Input/NextIteratorGetNextEstimator/Infer/Model/Iterator*
output_types
2*&
output_shapes
:���������
P
Estimator/Infer/Model/ShapeConst*
valueB"����   *
dtype0
�
#Estimator/Infer/Model/Input/FlattenReshape Estimator/Infer/Model/Input/NextEstimator/Infer/Model/Shape*
T0*
Tshape0
�
"Estimator/Infer/Model/ReadVariableReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
$Estimator/Infer/Model/ReadVariable_1ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
#Estimator/Infer/Model/Linear/MatMulMatMul#Estimator/Infer/Model/Input/Flatten"Estimator/Infer/Model/ReadVariable*
T0*
transpose_a( *
transpose_b( 
�
$Estimator/Infer/Model/Linear/AddBiasBiasAdd#Estimator/Infer/Model/Linear/MatMul$Estimator/Infer/Model/ReadVariable_1*
T0*
data_formatNHWC
c
,Estimator/Infer/Model/ReLU/ReLU/PositivePartRelu$Estimator/Infer/Model/Linear/AddBias*
T0
W
!Estimator/Infer/Model/ReLU/NegateNeg$Estimator/Infer/Model/Linear/AddBias*
T0
`
,Estimator/Infer/Model/ReLU/ReLU/NegativePartRelu!Estimator/Infer/Model/ReLU/Negate*
T0
P
#Estimator/Infer/Model/ReLU/ConstantConst*
valueB
 "���=*
dtype0
�
#Estimator/Infer/Model/ReLU/MultiplyMul#Estimator/Infer/Model/ReLU/Constant,Estimator/Infer/Model/ReLU/ReLU/NegativePart*
T0
�
#Estimator/Infer/Model/ReLU/SubtractSub,Estimator/Infer/Model/ReLU/ReLU/PositivePart#Estimator/Infer/Model/ReLU/Multiply*
T0
�
$Estimator/Infer/Model/ReadVariable_2ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
$Estimator/Infer/Model/ReadVariable_3ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
%Estimator/Infer/Model/Linear_1/MatMulMatMul#Estimator/Infer/Model/ReLU/Subtract$Estimator/Infer/Model/ReadVariable_2*
transpose_a( *
transpose_b( *
T0
�
&Estimator/Infer/Model/Linear_1/AddBiasBiasAdd%Estimator/Infer/Model/Linear_1/MatMul$Estimator/Infer/Model/ReadVariable_3*
T0*
data_formatNHWC
g
.Estimator/Infer/Model/ReLU_1/ReLU/PositivePartRelu&Estimator/Infer/Model/Linear_1/AddBias*
T0
[
#Estimator/Infer/Model/ReLU_1/NegateNeg&Estimator/Infer/Model/Linear_1/AddBias*
T0
d
.Estimator/Infer/Model/ReLU_1/ReLU/NegativePartRelu#Estimator/Infer/Model/ReLU_1/Negate*
T0
R
%Estimator/Infer/Model/ReLU_1/ConstantConst*
valueB
 "���=*
dtype0
�
%Estimator/Infer/Model/ReLU_1/MultiplyMul%Estimator/Infer/Model/ReLU_1/Constant.Estimator/Infer/Model/ReLU_1/ReLU/NegativePart*
T0
�
%Estimator/Infer/Model/ReLU_1/SubtractSub.Estimator/Infer/Model/ReLU_1/ReLU/PositivePart%Estimator/Infer/Model/ReLU_1/Multiply*
T0
�
$Estimator/Infer/Model/ReadVariable_4ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
$Estimator/Infer/Model/ReadVariable_5ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
%Estimator/Infer/Model/Linear_2/MatMulMatMul%Estimator/Infer/Model/ReLU_1/Subtract$Estimator/Infer/Model/ReadVariable_4*
transpose_a( *
transpose_b( *
T0
�
&Estimator/Infer/Model/Linear_2/AddBiasBiasAdd%Estimator/Infer/Model/Linear_2/MatMul$Estimator/Infer/Model/ReadVariable_5*
data_formatNHWC*
T0
�
!Estimator/Evaluate/Model/IteratorIterator*
	container *
output_types
2*
shared_name *5
output_shapes$
":���������:���������
�
-Estimator/Evaluate/Model/Input_Input/Zip/NextIteratorGetNext!Estimator/Evaluate/Model/Iterator*
output_types
2*5
output_shapes$
":���������:���������
S
Estimator/Evaluate/Model/ShapeConst*
valueB"����   *
dtype0
�
&Estimator/Evaluate/Model/Input/FlattenReshape-Estimator/Evaluate/Model/Input_Input/Zip/NextEstimator/Evaluate/Model/Shape*
T0*
Tshape0
�
%Estimator/Evaluate/Model/ReadVariableReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
'Estimator/Evaluate/Model/ReadVariable_1ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
&Estimator/Evaluate/Model/Linear/MatMulMatMul&Estimator/Evaluate/Model/Input/Flatten%Estimator/Evaluate/Model/ReadVariable*
T0*
transpose_a( *
transpose_b( 
�
'Estimator/Evaluate/Model/Linear/AddBiasBiasAdd&Estimator/Evaluate/Model/Linear/MatMul'Estimator/Evaluate/Model/ReadVariable_1*
T0*
data_formatNHWC
i
/Estimator/Evaluate/Model/ReLU/ReLU/PositivePartRelu'Estimator/Evaluate/Model/Linear/AddBias*
T0
]
$Estimator/Evaluate/Model/ReLU/NegateNeg'Estimator/Evaluate/Model/Linear/AddBias*
T0
f
/Estimator/Evaluate/Model/ReLU/ReLU/NegativePartRelu$Estimator/Evaluate/Model/ReLU/Negate*
T0
S
&Estimator/Evaluate/Model/ReLU/ConstantConst*
valueB
 "���=*
dtype0
�
&Estimator/Evaluate/Model/ReLU/MultiplyMul&Estimator/Evaluate/Model/ReLU/Constant/Estimator/Evaluate/Model/ReLU/ReLU/NegativePart*
T0
�
&Estimator/Evaluate/Model/ReLU/SubtractSub/Estimator/Evaluate/Model/ReLU/ReLU/PositivePart&Estimator/Evaluate/Model/ReLU/Multiply*
T0
�
'Estimator/Evaluate/Model/ReadVariable_2ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
'Estimator/Evaluate/Model/ReadVariable_3ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
(Estimator/Evaluate/Model/Linear_1/MatMulMatMul&Estimator/Evaluate/Model/ReLU/Subtract'Estimator/Evaluate/Model/ReadVariable_2*
T0*
transpose_a( *
transpose_b( 
�
)Estimator/Evaluate/Model/Linear_1/AddBiasBiasAdd(Estimator/Evaluate/Model/Linear_1/MatMul'Estimator/Evaluate/Model/ReadVariable_3*
data_formatNHWC*
T0
m
1Estimator/Evaluate/Model/ReLU_1/ReLU/PositivePartRelu)Estimator/Evaluate/Model/Linear_1/AddBias*
T0
a
&Estimator/Evaluate/Model/ReLU_1/NegateNeg)Estimator/Evaluate/Model/Linear_1/AddBias*
T0
j
1Estimator/Evaluate/Model/ReLU_1/ReLU/NegativePartRelu&Estimator/Evaluate/Model/ReLU_1/Negate*
T0
U
(Estimator/Evaluate/Model/ReLU_1/ConstantConst*
valueB
 "���=*
dtype0
�
(Estimator/Evaluate/Model/ReLU_1/MultiplyMul(Estimator/Evaluate/Model/ReLU_1/Constant1Estimator/Evaluate/Model/ReLU_1/ReLU/NegativePart*
T0
�
(Estimator/Evaluate/Model/ReLU_1/SubtractSub1Estimator/Evaluate/Model/ReLU_1/ReLU/PositivePart(Estimator/Evaluate/Model/ReLU_1/Multiply*
T0
�
'Estimator/Evaluate/Model/ReadVariable_4ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
'Estimator/Evaluate/Model/ReadVariable_5ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
(Estimator/Evaluate/Model/Linear_2/MatMulMatMul(Estimator/Evaluate/Model/ReLU_1/Subtract'Estimator/Evaluate/Model/ReadVariable_4*
T0*
transpose_a( *
transpose_b( 
�
)Estimator/Evaluate/Model/Linear_2/AddBiasBiasAdd(Estimator/Evaluate/Model/Linear_2/MatMul'Estimator/Evaluate/Model/ReadVariable_5*
data_formatNHWC*
T0
m
	eval_stepVarHandleOp*
shared_name	eval_step*
_class
 *
dtype0	*
	container *
shape: 
�
Beval_step/Initializer/eval_step/Initializer/ZerosInitializer/ShapeConst*
valueB *
_class
loc:@eval_step*
dtype0
�
Keval_step/Initializer/eval_step/Initializer/ZerosInitializer/Zeros/ConstantConst*
valueB	 "        *
_class
loc:@eval_step*
dtype0	
�
Geval_step/Initializer/eval_step/Initializer/ZerosInitializer/Zeros/FillFillBeval_step/Initializer/eval_step/Initializer/ZerosInitializer/ShapeKeval_step/Initializer/eval_step/Initializer/ZerosInitializer/Zeros/Constant*
T0	*

index_type0*
_class
loc:@eval_step
�
eval_step/InitializationAssignAssignVariableOp	eval_stepGeval_step/Initializer/eval_step/Initializer/ZerosInitializer/Zeros/Fill*
dtype0	
c
eval_step/Read/ReadVariableReadVariableOp	eval_step*
_class
loc:@eval_step*
dtype0	
8
eval_step/IsInitializedVarIsInitializedOp	eval_step
h
 eval_step/eval_step/ReadVariableReadVariableOp	eval_step*
_class
loc:@eval_step*
dtype0	
L
Estimator/Evaluate/ConstantConst*
valueB	 "       *
dtype0	
h
Estimator/Evaluate/AssignAddAssignAddVariableOp	eval_stepEstimator/Evaluate/Constant*
dtype0	
g
Estimator/Evaluate/AssignAdd_1ReadVariableOp	eval_step^Estimator/Evaluate/AssignAdd*
dtype0	
A
Estimator/Evaluate/GroupNoOp^Estimator/Evaluate/AssignAdd_1
b
Estimator/ReadVariableReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
J
Saver/ConstantConst*
_class
 *
valueB Bmodel*
dtype0
�
Saver/ReadVariableReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/ReadVariable_1ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/ReadVariable_2ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/ReadVariable_3ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/ReadVariable_4ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
Saver/ReadVariable_5ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
Saver/ReadVariable_6ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/ReadVariable_7ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/ReadVariable_8ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
Saver/ReadVariable_9ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
a
Saver/ReadVariable_10ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
a
Saver/ReadVariable_11ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
�
Saver/ReadVariable_12ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
Saver/ReadVariable_13ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
c
Saver/ReadVariable_14ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
c
Saver/ReadVariable_15ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
h
Saver/Constant_1Const*@
value7B5 B/_temp_fa9f4401-8f8c-4b4d-9131-e82ab6e86d17/part*
dtype0
Z
Saver/StringJoin
StringJoinSaver/ConstantSaver/Constant_1*
N*
	separator 
L
Saver/Constant_2Const"/device:CPU:0*
valueB
 "    *
dtype0
L
Saver/Constant_3Const"/device:CPU:0*
valueB
 "   *
dtype0
{
Saver/ShardedFilenameShardedFilenameSaver/StringJoinSaver/Constant_2Saver/Constant_3"/device:CPU:0*
_class
 
�
Saver/ReadVariable_16ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/ReadVariable_17ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
a
Saver/ReadVariable_18ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
�
Saver/ReadVariable_19ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
Saver/ReadVariable_20ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
Saver/ReadVariable_21ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/ReadVariable_22ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
c
Saver/ReadVariable_23ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
�
Saver/Constant_4Const"/device:CPU:0*�
value�B�Bglobal_stepB%Input/Flatten/OutputLayer/Linear/BiasB=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/BiasB@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/WeightsBglobal_epochB(Input/Flatten/OutputLayer/Linear/WeightsBYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/BiasB\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
Z
Saver/Constant_5Const"/device:CPU:0*#
valueBB B B B B B B B *
dtype0
�

Saver/SaveSaveV2Saver/ShardedFilenameSaver/Constant_4Saver/Constant_5Saver/ReadVariable_18Saver/ReadVariable_20Saver/ReadVariable_16Saver/ReadVariable_22Saver/ReadVariable_23Saver/ReadVariable_19Saver/ReadVariable_17Saver/ReadVariable_21"/device:CPU:0*
dtypes

2		
�
/Saver/WithControlDependencies/Identity/IdentityIdentitySaver/ShardedFilename^Saver/Save"/device:CPU:0*
T0*(
_class
loc:@Saver/ShardedFilename
}
Saver/ShapeConst0^Saver/WithControlDependencies/Identity/Identity"/device:CPU:0*
valueB"   *
dtype0
b
Saver/ReshapeReshapeSaver/ShardedFilenameSaver/Shape"/device:CPU:0*
T0*
Tshape0
s
Saver/MergeV2CheckpointsMergeV2CheckpointsSaver/ReshapeSaver/Constant"/device:CPU:0*
delete_old_dirs(
�
1Saver/WithControlDependencies_1/Identity/IdentityIdentitySaver/Constant^Saver/MergeV2Checkpoints0^Saver/WithControlDependencies/Identity/Identity*
T0*!
_class
loc:@Saver/Constant
�
Saver/ReadVariable_24ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/ReadVariable_25ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/ReadVariable_26ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/Constant_6Const"/device:CPU:0*q
valuehBfB\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
L
Saver/Constant_7Const"/device:CPU:0*
valueB
B *
dtype0
n
Saver/Restore	RestoreV2Saver/ConstantSaver/Constant_6Saver/Constant_7"/device:CPU:0*
dtypes
2
J
Saver/Identity/IdentityIdentitySaver/Restore"/device:CPU:0*
T0
�
Saver/AssignVariableAssignVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/WeightsSaver/Identity/Identity"/device:CPU:0*
dtype0
�
Saver/ReadVariable_27ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/ReadVariable_28ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/ReadVariable_29ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/Constant_8Const"/device:CPU:0*R
valueIBGB=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
L
Saver/Constant_9Const"/device:CPU:0*
valueB
B *
dtype0
p
Saver/Restore_1	RestoreV2Saver/ConstantSaver/Constant_8Saver/Constant_9"/device:CPU:0*
dtypes
2
N
Saver/Identity_1/IdentityIdentitySaver/Restore_1"/device:CPU:0*
T0
�
Saver/AssignVariable_1AssignVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/BiasSaver/Identity_1/Identity"/device:CPU:0*
dtype0
�
Saver/ReadVariable_30ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
Saver/ReadVariable_31ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
Saver/ReadVariable_32ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
r
Saver/Constant_10Const"/device:CPU:0*:
value1B/B%Input/Flatten/OutputLayer/Linear/Bias*
dtype0
M
Saver/Constant_11Const"/device:CPU:0*
valueB
B *
dtype0
r
Saver/Restore_2	RestoreV2Saver/ConstantSaver/Constant_10Saver/Constant_11"/device:CPU:0*
dtypes
2
N
Saver/Identity_2/IdentityIdentitySaver/Restore_2"/device:CPU:0*
T0
�
Saver/AssignVariable_2AssignVariableOp%Input/Flatten/OutputLayer/Linear/BiasSaver/Identity_2/Identity"/device:CPU:0*
dtype0
�
Saver/ReadVariable_33ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/ReadVariable_34ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/ReadVariable_35ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/Constant_12Const"/device:CPU:0*U
valueLBJB@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
M
Saver/Constant_13Const"/device:CPU:0*
valueB
B *
dtype0
r
Saver/Restore_3	RestoreV2Saver/ConstantSaver/Constant_12Saver/Constant_13"/device:CPU:0*
dtypes
2
N
Saver/Identity_3/IdentityIdentitySaver/Restore_3"/device:CPU:0*
T0
�
Saver/AssignVariable_3AssignVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/WeightsSaver/Identity_3/Identity"/device:CPU:0*
dtype0
�
Saver/ReadVariable_36ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
Saver/ReadVariable_37ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
Saver/ReadVariable_38ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
Saver/Constant_14Const"/device:CPU:0*n
valueeBcBYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
M
Saver/Constant_15Const"/device:CPU:0*
valueB
B *
dtype0
r
Saver/Restore_4	RestoreV2Saver/ConstantSaver/Constant_14Saver/Constant_15"/device:CPU:0*
dtypes
2
N
Saver/Identity_4/IdentityIdentitySaver/Restore_4"/device:CPU:0*
T0
�
Saver/AssignVariable_4AssignVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/BiasSaver/Identity_4/Identity"/device:CPU:0*
dtype0
a
Saver/ReadVariable_39ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
a
Saver/ReadVariable_40ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
a
Saver/ReadVariable_41ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
X
Saver/Constant_16Const"/device:CPU:0* 
valueBBglobal_step*
dtype0
M
Saver/Constant_17Const"/device:CPU:0*
valueB
B *
dtype0
r
Saver/Restore_5	RestoreV2Saver/ConstantSaver/Constant_16Saver/Constant_17"/device:CPU:0*
dtypes
2	
N
Saver/Identity_5/IdentityIdentitySaver/Restore_5"/device:CPU:0*
T0	
n
Saver/AssignVariable_5AssignVariableOpglobal_stepSaver/Identity_5/Identity"/device:CPU:0*
dtype0	
�
Saver/ReadVariable_42ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
Saver/ReadVariable_43ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
Saver/ReadVariable_44ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
u
Saver/Constant_18Const"/device:CPU:0*=
value4B2B(Input/Flatten/OutputLayer/Linear/Weights*
dtype0
M
Saver/Constant_19Const"/device:CPU:0*
valueB
B *
dtype0
r
Saver/Restore_6	RestoreV2Saver/ConstantSaver/Constant_18Saver/Constant_19"/device:CPU:0*
dtypes
2
N
Saver/Identity_6/IdentityIdentitySaver/Restore_6"/device:CPU:0*
T0
�
Saver/AssignVariable_6AssignVariableOp(Input/Flatten/OutputLayer/Linear/WeightsSaver/Identity_6/Identity"/device:CPU:0*
dtype0
c
Saver/ReadVariable_45ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
c
Saver/ReadVariable_46ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
c
Saver/ReadVariable_47ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
Y
Saver/Constant_20Const"/device:CPU:0*!
valueBBglobal_epoch*
dtype0
M
Saver/Constant_21Const"/device:CPU:0*
valueB
B *
dtype0
r
Saver/Restore_7	RestoreV2Saver/ConstantSaver/Constant_20Saver/Constant_21"/device:CPU:0*
dtypes
2	
N
Saver/Identity_7/IdentityIdentitySaver/Restore_7"/device:CPU:0*
T0	
o
Saver/AssignVariable_7AssignVariableOpglobal_epochSaver/Identity_7/Identity"/device:CPU:0*
dtype0	
�
Saver/GroupNoOp^Saver/AssignVariable^Saver/AssignVariable_1^Saver/AssignVariable_2^Saver/AssignVariable_3^Saver/AssignVariable_4^Saver/AssignVariable_5^Saver/AssignVariable_6^Saver/AssignVariable_7"/device:CPU:0
2
Saver/Group_1NoOp^Saver/Group"/device:CPU:0
X
ReadVariableReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
Z
ReadVariable_1ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
\
ReadVariable_2ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
Z
ReadVariable_3ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
�
UninitializedVariables/StackPackeval_step/IsInitializedglobal_step/IsInitialized3Input/Flatten/OutputLayer/Linear/Bias/IsInitializedgInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/IsInitializedNInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/IsInitializedjInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/IsInitialized6Input/Flatten/OutputLayer/Linear/Weights/IsInitializedglobal_epoch/IsInitializedKInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/IsInitialized"/device:CPU:0*

axis *
N	*
T0

\
!UninitializedVariables/LogicalNot
LogicalNotUninitializedVariables/Stack"/device:CPU:0
�
UninitializedVariables/ConstantConst"/device:CPU:0*�
value�B�	Beval_step:0Bglobal_epoch:0B[Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias:0B*Input/Flatten/OutputLayer/Linear/Weights:0Bglobal_step:0B^Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights:0B?Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias:0BBInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights:0B'Input/Flatten/OutputLayer/Linear/Bias:0*
dtype0
h
(UninitializedVariables/BooleanMask/ShapeConst"/device:CPU:0*
valueB"	   *
dtype0
g
+UninitializedVariables/BooleanMask/ConstantConst"/device:CPU:0*
valueB
 "    *
dtype0
�
(UninitializedVariables/BooleanMask/StackPack+UninitializedVariables/BooleanMask/Constant"/device:CPU:0*
T0*

axis *
N
i
-UninitializedVariables/BooleanMask/Constant_1Const"/device:CPU:0*
valueB
 "   *
dtype0
�
*UninitializedVariables/BooleanMask/Stack_1Pack-UninitializedVariables/BooleanMask/Constant_1"/device:CPU:0*
T0*

axis *
N
i
-UninitializedVariables/BooleanMask/Constant_2Const"/device:CPU:0*
valueB
 "   *
dtype0
�
*UninitializedVariables/BooleanMask/Stack_2Pack-UninitializedVariables/BooleanMask/Constant_2"/device:CPU:0*
T0*

axis *
N
q
1UninitializedVariables/BooleanMask/OnesLike/ShapeConst"/device:CPU:0*
valueB"   *
dtype0
u
9UninitializedVariables/BooleanMask/OnesLike/Ones/ConstantConst"/device:CPU:0*
valueB
 "   *
dtype0
�
5UninitializedVariables/BooleanMask/OnesLike/Ones/FillFill1UninitializedVariables/BooleanMask/OnesLike/Shape9UninitializedVariables/BooleanMask/OnesLike/Ones/Constant"/device:CPU:0*

index_type0*
T0
�
/UninitializedVariables/BooleanMask/StridedSliceStridedSlice(UninitializedVariables/BooleanMask/Shape(UninitializedVariables/BooleanMask/Stack*UninitializedVariables/BooleanMask/Stack_15UninitializedVariables/BooleanMask/OnesLike/Ones/Fill"/device:CPU:0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
i
-UninitializedVariables/BooleanMask/Constant_3Const"/device:CPU:0*
valueB
 "    *
dtype0
�
*UninitializedVariables/BooleanMask/Stack_3Pack-UninitializedVariables/BooleanMask/Constant_3"/device:CPU:0*

axis *
N*
T0
�
'UninitializedVariables/BooleanMask/ProdProd/UninitializedVariables/BooleanMask/StridedSlice*UninitializedVariables/BooleanMask/Stack_3"/device:CPU:0*

Tidx0*
	keep_dims( *
T0
j
*UninitializedVariables/BooleanMask/Shape_1Const"/device:CPU:0*
valueB"   *
dtype0
�
*UninitializedVariables/BooleanMask/ReshapeReshape'UninitializedVariables/BooleanMask/Prod*UninitializedVariables/BooleanMask/Shape_1"/device:CPU:0*
Tshape0*
T0
i
-UninitializedVariables/BooleanMask/Constant_4Const"/device:CPU:0*
valueB
 "   *
dtype0
�
*UninitializedVariables/BooleanMask/Stack_4Pack-UninitializedVariables/BooleanMask/Constant_4"/device:CPU:0*

axis *
N*
T0
i
-UninitializedVariables/BooleanMask/Constant_5Const"/device:CPU:0*
valueB
 "����*
dtype0
�
*UninitializedVariables/BooleanMask/Stack_5Pack-UninitializedVariables/BooleanMask/Constant_5"/device:CPU:0*

axis *
N*
T0
i
-UninitializedVariables/BooleanMask/Constant_6Const"/device:CPU:0*
valueB
 "   *
dtype0
�
*UninitializedVariables/BooleanMask/Stack_6Pack-UninitializedVariables/BooleanMask/Constant_6"/device:CPU:0*
T0*

axis *
N
s
3UninitializedVariables/BooleanMask/OnesLike_1/ShapeConst"/device:CPU:0*
valueB"   *
dtype0
w
;UninitializedVariables/BooleanMask/OnesLike_1/Ones/ConstantConst"/device:CPU:0*
valueB
 "   *
dtype0
�
7UninitializedVariables/BooleanMask/OnesLike_1/Ones/FillFill3UninitializedVariables/BooleanMask/OnesLike_1/Shape;UninitializedVariables/BooleanMask/OnesLike_1/Ones/Constant"/device:CPU:0*
T0*

index_type0
�
1UninitializedVariables/BooleanMask/StridedSlice_1StridedSlice(UninitializedVariables/BooleanMask/Shape*UninitializedVariables/BooleanMask/Stack_4*UninitializedVariables/BooleanMask/Stack_57UninitializedVariables/BooleanMask/OnesLike_1/Ones/Fill"/device:CPU:0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
i
-UninitializedVariables/BooleanMask/Constant_7Const"/device:CPU:0*
valueB
 "    *
dtype0
�
.UninitializedVariables/BooleanMask/ConcatenateConcatV2*UninitializedVariables/BooleanMask/Reshape1UninitializedVariables/BooleanMask/StridedSlice_1-UninitializedVariables/BooleanMask/Constant_7"/device:CPU:0*
T0*
N*

Tidx0
�
,UninitializedVariables/BooleanMask/Reshape_1ReshapeUninitializedVariables/Constant.UninitializedVariables/BooleanMask/Concatenate"/device:CPU:0*
T0*
Tshape0
i
-UninitializedVariables/BooleanMask/Constant_8Const"/device:CPU:0*
valueB
 "����*
dtype0
�
*UninitializedVariables/BooleanMask/Stack_7Pack-UninitializedVariables/BooleanMask/Constant_8"/device:CPU:0*

axis *
N*
T0
�
,UninitializedVariables/BooleanMask/Reshape_2Reshape!UninitializedVariables/LogicalNot*UninitializedVariables/BooleanMask/Stack_7"/device:CPU:0*
Tshape0*
T0

w
(UninitializedVariables/BooleanMask/WhereWhere,UninitializedVariables/BooleanMask/Reshape_2"/device:CPU:0*
T0

�
*UninitializedVariables/BooleanMask/SqueezeSqueeze(UninitializedVariables/BooleanMask/Where"/device:CPU:0*
squeeze_dims
*
T0	
i
-UninitializedVariables/BooleanMask/Constant_9Const"/device:CPU:0*
valueB
 "    *
dtype0
�
)UninitializedVariables/BooleanMask/GatherGatherV2,UninitializedVariables/BooleanMask/Reshape_1*UninitializedVariables/BooleanMask/Squeeze-UninitializedVariables/BooleanMask/Constant_9"/device:CPU:0*
Taxis0*
Tindices0	*
Tparams0
W
UninitializedResources/ConstantConst"/device:CPU:0*
valueB *
dtype0
5
ConstantConst*
valueB
 "    *
dtype0
�
ConcatenateConcatV2)UninitializedVariables/BooleanMask/GatherUninitializedResources/ConstantConstant*

Tidx0*
T0*
N
�
UninitializedVariables_1/StackPackglobal_step/IsInitialized3Input/Flatten/OutputLayer/Linear/Bias/IsInitializedgInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/IsInitializedNInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/IsInitializedjInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/IsInitialized6Input/Flatten/OutputLayer/Linear/Weights/IsInitializedglobal_epoch/IsInitializedKInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/IsInitialized"/device:CPU:0*
T0
*

axis *
N
`
#UninitializedVariables_1/LogicalNot
LogicalNotUninitializedVariables_1/Stack"/device:CPU:0
�
!UninitializedVariables_1/ConstantConst"/device:CPU:0*�
value�B�Bglobal_epoch:0B[Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias:0B*Input/Flatten/OutputLayer/Linear/Weights:0Bglobal_step:0B^Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights:0B?Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias:0BBInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights:0B'Input/Flatten/OutputLayer/Linear/Bias:0*
dtype0
j
*UninitializedVariables_1/BooleanMask/ShapeConst"/device:CPU:0*
valueB"   *
dtype0
i
-UninitializedVariables_1/BooleanMask/ConstantConst"/device:CPU:0*
valueB
 "    *
dtype0
�
*UninitializedVariables_1/BooleanMask/StackPack-UninitializedVariables_1/BooleanMask/Constant"/device:CPU:0*
T0*

axis *
N
k
/UninitializedVariables_1/BooleanMask/Constant_1Const"/device:CPU:0*
valueB
 "   *
dtype0
�
,UninitializedVariables_1/BooleanMask/Stack_1Pack/UninitializedVariables_1/BooleanMask/Constant_1"/device:CPU:0*

axis *
N*
T0
k
/UninitializedVariables_1/BooleanMask/Constant_2Const"/device:CPU:0*
valueB
 "   *
dtype0
�
,UninitializedVariables_1/BooleanMask/Stack_2Pack/UninitializedVariables_1/BooleanMask/Constant_2"/device:CPU:0*
T0*

axis *
N
s
3UninitializedVariables_1/BooleanMask/OnesLike/ShapeConst"/device:CPU:0*
valueB"   *
dtype0
w
;UninitializedVariables_1/BooleanMask/OnesLike/Ones/ConstantConst"/device:CPU:0*
valueB
 "   *
dtype0
�
7UninitializedVariables_1/BooleanMask/OnesLike/Ones/FillFill3UninitializedVariables_1/BooleanMask/OnesLike/Shape;UninitializedVariables_1/BooleanMask/OnesLike/Ones/Constant"/device:CPU:0*
T0*

index_type0
�
1UninitializedVariables_1/BooleanMask/StridedSliceStridedSlice*UninitializedVariables_1/BooleanMask/Shape*UninitializedVariables_1/BooleanMask/Stack,UninitializedVariables_1/BooleanMask/Stack_17UninitializedVariables_1/BooleanMask/OnesLike/Ones/Fill"/device:CPU:0*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
k
/UninitializedVariables_1/BooleanMask/Constant_3Const"/device:CPU:0*
valueB
 "    *
dtype0
�
,UninitializedVariables_1/BooleanMask/Stack_3Pack/UninitializedVariables_1/BooleanMask/Constant_3"/device:CPU:0*
T0*

axis *
N
�
)UninitializedVariables_1/BooleanMask/ProdProd1UninitializedVariables_1/BooleanMask/StridedSlice,UninitializedVariables_1/BooleanMask/Stack_3"/device:CPU:0*

Tidx0*
	keep_dims( *
T0
l
,UninitializedVariables_1/BooleanMask/Shape_1Const"/device:CPU:0*
valueB"   *
dtype0
�
,UninitializedVariables_1/BooleanMask/ReshapeReshape)UninitializedVariables_1/BooleanMask/Prod,UninitializedVariables_1/BooleanMask/Shape_1"/device:CPU:0*
T0*
Tshape0
k
/UninitializedVariables_1/BooleanMask/Constant_4Const"/device:CPU:0*
valueB
 "   *
dtype0
�
,UninitializedVariables_1/BooleanMask/Stack_4Pack/UninitializedVariables_1/BooleanMask/Constant_4"/device:CPU:0*

axis *
N*
T0
k
/UninitializedVariables_1/BooleanMask/Constant_5Const"/device:CPU:0*
valueB
 "����*
dtype0
�
,UninitializedVariables_1/BooleanMask/Stack_5Pack/UninitializedVariables_1/BooleanMask/Constant_5"/device:CPU:0*
T0*

axis *
N
k
/UninitializedVariables_1/BooleanMask/Constant_6Const"/device:CPU:0*
valueB
 "   *
dtype0
�
,UninitializedVariables_1/BooleanMask/Stack_6Pack/UninitializedVariables_1/BooleanMask/Constant_6"/device:CPU:0*

axis *
N*
T0
u
5UninitializedVariables_1/BooleanMask/OnesLike_1/ShapeConst"/device:CPU:0*
valueB"   *
dtype0
y
=UninitializedVariables_1/BooleanMask/OnesLike_1/Ones/ConstantConst"/device:CPU:0*
valueB
 "   *
dtype0
�
9UninitializedVariables_1/BooleanMask/OnesLike_1/Ones/FillFill5UninitializedVariables_1/BooleanMask/OnesLike_1/Shape=UninitializedVariables_1/BooleanMask/OnesLike_1/Ones/Constant"/device:CPU:0*

index_type0*
T0
�
3UninitializedVariables_1/BooleanMask/StridedSlice_1StridedSlice*UninitializedVariables_1/BooleanMask/Shape,UninitializedVariables_1/BooleanMask/Stack_4,UninitializedVariables_1/BooleanMask/Stack_59UninitializedVariables_1/BooleanMask/OnesLike_1/Ones/Fill"/device:CPU:0*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask
k
/UninitializedVariables_1/BooleanMask/Constant_7Const"/device:CPU:0*
valueB
 "    *
dtype0
�
0UninitializedVariables_1/BooleanMask/ConcatenateConcatV2,UninitializedVariables_1/BooleanMask/Reshape3UninitializedVariables_1/BooleanMask/StridedSlice_1/UninitializedVariables_1/BooleanMask/Constant_7"/device:CPU:0*
T0*
N*

Tidx0
�
.UninitializedVariables_1/BooleanMask/Reshape_1Reshape!UninitializedVariables_1/Constant0UninitializedVariables_1/BooleanMask/Concatenate"/device:CPU:0*
Tshape0*
T0
k
/UninitializedVariables_1/BooleanMask/Constant_8Const"/device:CPU:0*
valueB
 "����*
dtype0
�
,UninitializedVariables_1/BooleanMask/Stack_7Pack/UninitializedVariables_1/BooleanMask/Constant_8"/device:CPU:0*
T0*

axis *
N
�
.UninitializedVariables_1/BooleanMask/Reshape_2Reshape#UninitializedVariables_1/LogicalNot,UninitializedVariables_1/BooleanMask/Stack_7"/device:CPU:0*
Tshape0*
T0

{
*UninitializedVariables_1/BooleanMask/WhereWhere.UninitializedVariables_1/BooleanMask/Reshape_2"/device:CPU:0*
T0

�
,UninitializedVariables_1/BooleanMask/SqueezeSqueeze*UninitializedVariables_1/BooleanMask/Where"/device:CPU:0*
T0	*
squeeze_dims

k
/UninitializedVariables_1/BooleanMask/Constant_9Const"/device:CPU:0*
valueB
 "    *
dtype0
�
+UninitializedVariables_1/BooleanMask/GatherGatherV2.UninitializedVariables_1/BooleanMask/Reshape_1,UninitializedVariables_1/BooleanMask/Squeeze/UninitializedVariables_1/BooleanMask/Constant_9"/device:CPU:0*
Taxis0*
Tindices0	*
Tparams0
�
Initializers/Variables/GlobalNoOpo^Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/InitializationAssignr^Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/InitializationAssignS^Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/InitializationAssignV^Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/InitializationAssign;^Input/Flatten/OutputLayer/Linear/Bias/InitializationAssign>^Input/Flatten/OutputLayer/Linear/Weights/InitializationAssign"^global_epoch/InitializationAssign!^global_step/InitializationAssign
%
Initializers/Resources/SharedNoOp
M
GroupNoOp^Initializers/Resources/Shared^Initializers/Variables/Global
E
Initializers/Variables/LocalNoOp^eval_step/InitializationAssign
$
Initializers/Resources/LocalNoOp
"
Initializers/Lookup/TablesNoOp
j
Group_1NoOp^Initializers/Lookup/Tables^Initializers/Resources/Local^Initializers/Variables/Local
��
%train_dataset/TensorToOutput/ConstantConst*��
value�B�	�	"Էk��?��n N?�٣��y�?�}�?ɓ?O%F?�%m���'?���>����NY�?�����ϛ?��+>E���d|�dAt?@��>m�G�uyE�l��>벪��ޗ>���}�!����f�!V��쮾0Г?Y1�?@��?�-?���>��6�@�E���d|�O%F?�%m��Ȼ��ξ�@3
۾W�?��[�:o��˝��E��g��==֗����4	�cZ����%?l��w4˾��+>E��bm?J�>=֗�u�_��ĝ��G?�C�?�$��٣�h~¾E��*�M�����'����ϾMF���G?��o���?N�G��$�@a>�>������?¬�>��>ʏ>�@< ����:o��E���U�`���|���ͫ�uyE���1����?�-�K��܍>lk��bm?����Z���w뿽�'��)*>o,���?N�G�xB>E���Q]��C+��@;���W���t�)P=?��_?�$���x���E����?c����ӯ�_3_�a�?z;��>h<U?J��E���d|� ��?E���W�y>����a:@�$��O4��]Q����?�Q�?Hh�϶1@u�_�(){?�տ Ý>Y;y�K����ԡ�>"��gÀ�7ٛ?w뿽_3_�����4���3?�$�������R�?2��>_�v��\6>�͛?c�c?cZ����N�&?�$��T'F��������ڎ׿3��=7ʏ>ʇ@a�?U�5���?%�M���r�*�M��?�4����w�l�p=ڄ��/��>�7_��O4�͓\�D���y+�Sv��wm�?��g��j�?k����>Y;y��S�?��r�!���M�><�>��w�	�����1���B?�����$��T'F�r��ى�@S��x>��<��@��>�h�<W�?K���y�?r�A�&��m�>Z���hݏ���t��������̢��z� �f��!���"��%K8�@��>V�7�l�p=����Զ����{>g�@�:�r�!-@/
��|��(��(){?(.��xua����K��v������>�=�M�>*�?u�_��?�����%?JJ����x��h����?	(@5u�Z���У�MF����> Ý>���XD_>:o���q(��d�?��?z[	�����-��C��w���ޗ>%�M�������>�m��]�P���x>��7?�j�?)*>l�R����>�W�>�S��F�?M�@\�>�@;�(����t�1����̢��#Xo�����Q>A���uG��=t@u�_�ʇ@@��?	�οW�?�$��:o�������'�?c���hݏ��4	�B���=����?%�M��.>0iG?2��>�u?���7ʏ>���>^�N���G>�!��N�G�����?�r��M��˴?���>k����_?�-���	�Q�D���B�h�]V?7ٛ?7ʏ>MF���{�Y��?�!���G*�������?��;=]�P��z�?(����?�]�?Զ��K�?XD_>f��E���>>����?�쮾x�?�R4?�ۚ��W?��G>z� �;j<lk��9
?g��=�+u?�W?�'��^�N��J�=����W�>�h��}Y��f� ��?��@ ��?��t��߰�[:=��28��$��ۺ/��M7���Q}ӿj��?���=
��?�C�r�>�#������=�q(��t�p�X���x>��O?���=����G>ȢQ�	�Q����?۲>�+?3��=V�7��?�2�"���������@�@=����Q]��O[=���>��>�R4?���U�5�1��?'}�?	�g���?]+�?t�]�]��?��<S�:@�pI@�3N�[�@�W�>��=r��t�wd|?�O���?���?FwоI}'��ޗ>%�M�	�Q�r��y6�k���|�����R4?P6�H�?��ན�-?�q~��R�?Q��?U���G�K��Ȼ�D������ƀ>JJ��N�G���;���a����d��a�}�V�7��?���R=��o���{>��x��S����?�|?��7��|�����>O?���o,���@?�G*�����@�1䃾�z>`�?hݏ�l�p=P6���#[?���S����?Y'?�xj�@��>��<�R4?�ۚ�3
۾�M���O4���;˝��t�B��� "?(��D���>����������c�>!���A:�=k��<�>?�'�D���¦]�I}'���G>���@f��Sux?
9��K?<�>w뿽9Ӟ��,@��W�?'}�?�R�r� <~<E�5@�|����>��@�Y�?�4��#[?�G*��$�]?�r�?2=E=���=��-����/��>�!������+>A�_?���?��k���x>V�7�9Ӟ������h�<�ޗ>�$���$�0iG?/b=@��̽�|���ӯ�:]�TXi?�W?�$��$���h�r��Ѿ�M�>�������t�V��?��o���?��x�v����q(��y6��B�z[	���'?���?��> Ý>���w4˾���>��@�aF�/� ?Z������>uyE�^�N��OQ?b����٣��$���@��Ѿ�h?B�@f��>uyE�FwоI}'�1+��W�>�����>��ؽ��7�2=E=��l�p=^�N�K��������[�h~¾���?1䃾����2=E=��c�c?ڄ����o�QԾ��?:o��lk��B�h�5u�P��?��W���@k����B?������:o��!���"�I�'�?*�?(����t��{�K���9l���x�}��q(��Q]��⛿Ӥ2?�_>��K?2�"��u?�28�w4˾�q~�D���*�M��q-�E�V�7�9Ӟ�r��@�-?W�?r��?��+>����y6��\Y@<#�x�?�?�]�?�w����?��h?���?r�J�l�\�>G�K�7ʏ>��K?�?�h�<W�?G�?w�+?����e����?W�ɿ(��Y1�?)P=?0�?�4E� ��>J���F�?
�U?.��?�|��w뿽�4	���p������?�����@���=�Ǔ>g��=�⿢�)@D�������"�?����%�M��S�D���	�.�_O(?<�>�қ�uyE�Fwо����d/�XD_>
��>��?B�M?�(��%m���_3_�¦]�������w4˾�q~���?�w�>�$C��\6>�ӯ��4	������3N������G*���ؾlk���ѾJz�`�?��Ͼl��?(.���ƀ>̢����!�����ى�\�>�+u?��>:]���1��w���l.=w4˾J��Չ�?Y'?!&�=�d?p׋���t����z;����>�J0@,H�����=���?'�?��x>�g?l�p=2�"�`yD�#[?��=ZW�]?��;=�s��|��0Г?(){?�@������?��=�R����>9
?� �?�Ȱ�hݏ�l�p=P6�`yD��>K��v������?���?��i>�� ?hݏ�	����ۚ�K����?%�M��h�r��d|�=!�*�?�_>O?��K?h��N�&?N�G��۠��M7�aF����c�����g���>¦]���%?�!���G*�:o���M7�"����.�@��>���4	��C��OQ?< ��h<U?UX�>r�*�M��z>ma?��9Ӟ�vׄ�`yD��X�]B�@r�K����?��?]�P�:�S??�'��j�?�G?ؒ;?��q�w4˾xB>E��ဲ�G�>'0о�Ȼ�uyE����o,�96R�N�G��,;�g��B�h�/�=Ӥ2?��uyE�2�"�"�?̢��%�M��q~�)N<�{��z>:�S?��W��'��ʏ>o,��>��>��0iG?�@O%F?�쮾hݏ�:]�ʏ>G��>QԾ��?�.>�F�?!�.?gd?-����w��ξvׄ� Ý>������x�xB>˝�������j��?u�_�(){?2�"� Ý>�!���G*��h��}Y���Ǜ���x>��W��ξ����0���>�G*��]Q����ŀ��'D^�ma?w뿽O?)P=?0�?�4E� ��>J���F�?
�U?.��?�|��w뿽�4	�ʏ>�0����?w4˾�܍>ԡ�>�f��q-��4���_>�R4?lf@z;�W�?��x��@=E���d|����?G�K��ͫ�uyE�����r�>����z� �h~¾�}Y��e������쮾w뿽��t��?��o����?��x�=8�?�}Y��Q]�wd|?���!@��t���?�����h??@�۠�E�����&��>z[	�Ճ?�r�?��1��0�����?�٣�v������=A�&�<>?<�>���>�>cZ����?�7_��٣��۠�A�_?2��>�B�E���?��>%Tl�s�&��?�٣����0iG?aC+>�s��\6>f��>�R4?^�N�o,������=�S�����m��q=�+u?��w�O?k���OQ?�4E��O4��B
@E��aF�lȾ�� ?V�7�:]�^�N�o,������=�S�����m��q=�+u?��w�O?¦]���>�7_�z� �}��R�?�VJ>B��� "?���=��-��{���G>1+��$�����!���!������Ӥ2?u�_�Y1�?B����C�?d/����@=���aC+>/�=z[	���7?:]��߰�G��>Y;y��٣�,H��)N<	�.��ɿP��?��7?�?��d>�~�?�F���G*�����q(��{��n��Ȱ�?�'���t�2�"�xua���{>w4˾&�lk�����A��z[	�¬�>��K?�{����?�M�����@�۠���?�|?]V?E�V�7�D�����p��~�?����w4˾f���q(��>>�N㔾3��=��>��>)*>3
۾��G>��=�����>A:�=O%F?���> ��?MF��B����h�<������[�h~¾r��d|��h۾'0оw뿽�ξa�?[:=��u?G�?�.>˝�����KH=�Z���w뿽E9�?B���*ߩ?&D�=r��?:o��D���J�l�l���d?��>(){?�������̢��z� �f��!���"��%K8�@��>V�7�l�p=��? Ý>�M���W�>�S�lk���@�+?z[	���Ͼ:]�ad@z;���?��[��۠��.?�{��+?<#���>l�p=���=����?��[�-��@)N<
9��x>���\Ո@D���k��z;���?w4˾�����>B�M?\�>@��>���ĝ�2�"���B?�ޗ>%�M���ؾ�R�?���?�:��Z���ԽG?ʇ@l��>xua���{>�٣��������t�X���|���g?�R4?��?1�
�W�?�$�����.?� @&��>z[	�w뿽:]��m�?I}'���?XD_>xB>A�_?���?�f�?=֗�m�G��ξ���=��懡?K���m=�q(��y6��u�@��>���=�4	�V��?`yD��u?��x�:o���M7��f���;0ҿ0Г?c�c?2�"��q~���q���X�½��?�~]?/�=ma?��:]��K@�Ln?����K���i?˝�����q�M���x>�ӯ���-�Fwоz;�&D�=�G*���E���d|��i�2=E=u�_�(){?P6� Ý>�-�w4˾�7?˝�)c����-�����R4?I�?,h���?��?���>����y6�-�?��X�F��?9Ӟ�ʏ>��W�? ��>�R����=��?O%F?E��ӯ��?�I�E@C> ���@ ��>�R����=��ؽ�@0ҿ�g?�ξ�$�>��?]��>z� ��,v@E��R�>��C>���h�-@�'��wh@�4��}�@��x����r�!�� ��?��3�Ճ?�?�����z;�������[��*�lk��y+��,��@��>У�:]���1�$��?�#����x����g�Q]�n��� ?��˿ʇ@cZ���|?�X�w4˾q��>�M7���?j.�>G�K�7ʏ>��t����ƀ>JJ��N�G���;���a����d��a�}�V�7��?�2�"��S4?̢���G*��.>E��"��~��?՛@�g?��-�Fwо��_?̢��#Xo�����q(�*�M�l���%m���)@D����?`yD���?��w�+?r��d|�O%F?�|����?�?�2�"�xua���{>w4˾&�lk�����A��z[	�¬�>��K?�]�?�o��|��?K��ZW�r�����D�>3��=�ȫ?��>)*>��l���G*��U5��M7���!V���\��g?�R4?�K@�1�
����>�O4��@=�M7��m�������\6>(��D�����1��q~�&D�=%�M�xB>ԡ�>�VJ>�,��Ӥ2?}% �O?cZ���񱿻��?�W�>���>����>>���P?�� ?��>�>¦]���>�7_�z� �}��R�?�VJ>B��� "?���=��-���>��B?�+��W�>�h�E�����&��>E�?�'��4	��C���B?����w4˾f��r��y6��S?�R�?���?MF��^�N��g?����%�M����>]?�m�������+u?���>�m�? Ý>ۓ�?��x�v������ <~<��	@'0о��	���)*>�-?���w4˾X�½��@���j]??��X��ӯ��ξ�{�z;�������&����>��6Eڿ:�S??�'���?@��?U�5�N�&?%�M���ؾ@�ii> G�><�>Gˣ?l�p=�ۚ�����1+�K��ۺ/�lk��q
¾Fo���7�?��(){?^�N��3N��7_��G*�v���D���Y ����Ӥ2?��_3_�I�?�q~�W�?)4?h~¾����m��px�?2=E= ��?O?cZ����>����%�M���=���>۲>���?7ٛ?}% ��'���!@/��>W�?XD_>:o��E���U�cq�?c�����c�c?����k��?������x�f�����"��B ½'0о��w��?��������������٣������@�ii>��k���?��?y>��p��4����{>K��ZW��M7�Q���#"ҿ "?(���r�?�x?��>��G>��x���m=Sux?R�>cq�?��\���_3_��G?z;����>��f��r��e�8R#>Z�����7?���>�@l�R����?��=�S�Չ�?y0F?-�?�Ȱ�¬�>l�p=^�N�z;����N�G��R����>�=B�����x>ԽG?:]��K@���?Y;y����q~���@�	�.�'�?�7�?m�G�	���a�?�s��Q��?XD_>�$�r��e���7��4�����?Y1�?�{�B��?�!�����@�,;�r�5����n��z�?��w�E9�?vׄ��w��l��XD_>!���@�A���Y�Ŀ��*�(�����?l��>�񱿍u?z�@�i?˝�a���f��?��\��ӯ���>��?I}'�1��?%�M��U5�D���A�&�]�P��\6>�W?�)@��ƿ��B?������x���˝��ى��ȿ�=%@��ʇ@��R=�J?��?w4˾�3
?!���9o�?\�>a�}�V�7��ĝ�(.���ƀ>1+�S"�?h~¾�}�?B�M?�VԼ2=E=p׋�D���k����o��>��:o��Q>1䃾-0�<�>^Ƴ?S�:@^�N�r�>d/���x��q~����>"�I��L�� "?��Ͼ9Ӟ�lf@�0��W�?���>�,;����?�r�~��?�@;��g?_3_�T6��벪���q�z� �}�˝��ى�$�1��� ?V�7�"�4@��R=�OQ?QԾ��x���=g���~]?�t�>�@;���ϾD���^�N�U�5��$�%�M�X�½�F�?
9�����쮾���ξcZ����?�7_��٣��۠�A�_?2��>�B�E���?��>��p�Ln?JJ���٣��$���?��7@�,��'0о��g�MF��>2����������G*�,H���.?۲>��]�Ӥ2?F��?�>�$�>z;�JJ����x�;j<���>B�h�Ds�=3��=�ҋ?�?TXi?�J�=��q�w4˾��?�}Y�2��>�q>>2����@_3_�B����3N����>����<@r�J�l��+?��\��ӯ�:]�P6�=���+�w4˾�۠�r��Q]��z>E�hݏ�D���Ţ.?��>#[?�G*�,H�����>Y'?��?�+u?��:]�cZ����G>�X���-?	�Q��M7��d�?&��>��X�u�_���t�)*>o,���?N�G�xB>E���Q]��C+��@;���W���t���>NY�?1+���=�@=�q(��y6��Da?�� ?���?�l��>��%?�X����>��+>!���1䃾���?��X���Ͼ��-��$�>��?�T�<�|?�}�?Ƅ]@aS?@wd|?a�}�^Ƴ?D���ʏ>=��JJ��%�M�(��=��?}>�?�6J?G�K���W�D���¦]��3N��F���G*�,H��!���	�.����� "?�ҋ?��K?����W?l��w4˾X�½���=��?`���쮾���?���d>��X?�7_��O4���m=����ى����>���}% ��GQ�P6��OQ?�>w4˾bN���@R�>B����쮾(��_3_�¦]�/��>�T�K��,;�Q>�y�?/�=ݰ�?���'���C�a�?Y;y�z� ��.>���
9�X���s񾲸w��?����=xua���{>��ZW�Sux?A0>迓�<�>���>���?�x?�4����?��=�R��M7��Ѿ e�?@��>�g?l�p=2�"�l�R��>���۠�Չ�?�6?q=Ӥ2?��Ͼ:]�Ţ.?xua�懡?w4˾;j<D���A0>s)��E����?>2����N�&?��[�ZW�T@J+f@��;�'0о��Ͼά?������G>�l.=z� �@��@)N<�Ǔ>/�=�w���y@�'���n>@�h�<z�(@���?�7?�.?bm?n�+@��\���7?,V@�C��u?����%�M�:o��˝�����+?]��?��W�D�����1�/��>������v���)N<q
¾�z>��?7ʏ>MF���?��o���?��x�=8�?r��Q]�wd|?��Q�@��t������4����?h<U?!�Q>�{�-`)��� ?��<O?2�"��ƀ>< ���G*��S�)N<Q���~�<�>�ӯ�L� �¦]��W?�4E�<�|?q��>)N<�l@g��=��X�hݏ�:]�k���|?����K���q~�g��q
¾'�?*�?(����t�@��?I}'�n N?��x�
��>g���m�����>a�}����=���>��>U�5�n N?�$����ؾ��?�r�;a���@;��ӯ�E9�?�a�?U�5�|��?�٣�KS?lk���=�+?����w?��-�P6���>< ��w4˾�@=D�����4��>@��>hݏ��ξ�@�q~�W�?��=�,;��}�?!�.?cq�?'����?�4	���>U�5��3?w4˾��$�lk��"�I��:��-���ӯ�E9�?(.����N�&?�$��,H����@��U��n�P��?¬�>�>�]�?G��>W�?�٣���@E��J�l�cq�?a�}���Xi�{�z;�����٣���ؾr��d|��(��� ?w뿽��t�2�"���B?�!���G*�	�Q�)N<�y�?q�>�˴?����-��ۚ���%?�9l�%�M��S�lk��y+�%K8�*�?�����?��@�4E���x�5��?�}Y��y6���7���x>}% �_3_��?z;��h?��=�q1?�R�?Y'?9>z[	����>��>��>l�R��3?�$��c�>r���������=֗���ά?��1���>�#���G*��$��}�?��ؽ�(�Ӥ2?��y>wh@`yD�W�?%�M��$���@�*�M���?W�ɿՃ?L� ���?�4�����>��x��,;�g��"�𾼼��3��=Gˣ?���?¦]���o���{>��!����ŀ��Ӌ�@��>?�'�ʇ@�]�?o,��h?w4˾�$�r�����?-����'?c�c?k��G��>�#��w4˾v���!���1䃾� �>�7�?}% ��?�^�N��~�?�T$��%Z@G~���;=�ߧ�ma?�ӯ�L� ���d>Ln?������?f��!����Ѿ��?@��>У���t�����?�7_���=f���q(�y+�&��>�� ?�ك���-�cZ��U�5���G>K��J����@�	�.�lȾ�s��_>�>�C�"�?�$�K�򾑎�>Q>���?���쮾��Ͼ�'����p�Զ���X�z� ��h�Q>
9�(���E�}% ��'����K?Ln?��{>%�M�;j<]?��?�u?E�u�_�9Ӟ���d>6]�?�?z� ����@D���ɓ?�	�������M@D���ʏ>�s����?�G*�bN����>y0F?@�1��� ?���>���?��1��?�?d/���UX�>ԡ�>���?�h۾G�K���w�:]��{��|?Y;y���="�>�M7��f���@�:�S?��_3_������?��@?K��(��=��@��y�?���z[	���Ͼ:]�Ţ.?6]�?�F��K��h�r���?�쮾(���>�{�z;��#��N�G�bN����=y+�]�P�ma?���?�¦]�z;�96R�N�G�:o��)N<�=��.�@��>�W?:]�����/��>��q�w4˾�$�Չ�?�"$@;�����*���W�MF��cZ��z9@�M����[����}Y�"��	���w���қ��ξ��1�G��>̢��%�M��S����=\�?���>�7�?�ӯ���-�)�$@��?ۓ�?�!@X�½�}Y��{�8'9@W�ɿ¬�>�ξ��1�Ln?����٣���ؾ���>�f��N>:�S?���=�?�>��w���������]Q��}�?qE ?1;.�B�@�W?^R@��K?r�>�+� ��>��ԡ�>�VJ>U@�?c���hݏ���t�k��?U�5��3?��x�f���}Y��e��Da?}j�����=�'���K@��g?����K��q��>�q(��>>��%���\6>�ӯ���-���p�z9@�!������$����ŀ���z>@��>O���ĝ�>2����?�7_�XD_>�S�!������h���?��<_3_�k����%?�+�x�@/�?Ƅ]@e�?�t�>��X���w�:]�����z;��4E�����?r�aF�lȾ��X��ȫ?�?����=,h���?���,H��(�2@}>�?�GK�3��=��'?���?�?�4��W�?��x�}�Q>"�I�&��>c�����g��?���p�1�
�JJ���O4�"�>����m�������%m� ��?:]����=����{>��>!��}Y�A�&�*�x��s�ͫ��4	�U#�o,�d/��٣��:���@��?Hh⿜=t@(��]=@3�?z;�XM@�@�܍>E���e�-�?-��u�_��R4?%Tl��OQ?̢��K��D��Sux?�{��z��� ?Ճ?��>ʏ>	�ο���>z� ��U5�!����f�tʽ�c�����Ͼ��K?��p� Ý>�X��G*��q~��.?Ռ�?�Z�=��*���:]���1�lݿW�?�����@��Q]���߿�� ?(��
��?�m�?�h�<ۓ�?_?ҧ<?r��d|�'�?
]ڿx�?(){?a�?��>1+�w4˾f��A�_?��;=��8?c���(����-�����벪����>��x�͓\����?�VJ>��k�Z���:�@
��?��p�1�
�JJ���O4�"�>����m�������%m� ��?:]�I�?3
۾1��?�W�>	�Q����> <~<px�?�쮾��5@O?B����o��]��>��[��q~���@��y6��M�>�쮾7ʏ>uyE���1���>�#��%�M��h�ԡ�>Q��?#��>wm�?��Ͼ�?�>2���w�������f�������1S��@��>�ͫ�9Ӟ���1���G>1+��O4�	�Q��}Y��Q]��޾Z���u�_��'��B����w��N�&?�$������0iG?��;=H�"��\6>Gˣ?O?�]�?I}'�v
�?��?�q~�]?�ii>-�?�4���û?MF����1�1�
��+���J�ｲ}�?A0>�gX�]��?w뿽�4	��{��e@������J��E��J�l�~�Կ7ٛ?�ͫ�ά?l��>lݿ�$��٣�	�Q�1Q�����Q�? "?�ȫ?L� ���?1�
�#[?�$��s�1@E��5���cq�?G�K���:]�����*ߩ?����G*�(��=lk���ii>\�>�쮾}% ��?��K@�Ln?���%�M�6ң=lk��aF�/
�<�>�ӯ���K?�K@���o�JJ���$����0iG?!�.?���ma?m�G�9Ӟ�TXi?�3N���?��A?�.>�q(���ؽU@�?E���	���P6����?�$�XD_>X�½���R�>q=��x>(���ξ��?���X���=6ң=Q>bm?�<�?�4����g���t�¦]���%?������>�G?lk���Ѿ%K8�:�S?����K?(.��z9@�����G*�v�?r�5���ѩ&�P��?��g����>��?�h�<|��?%�M�}�r�A�&����?'����Ͼ�'��Fwо벪�W�?��>(��=A�_?2��>��C>���hݏ�9Ӟ���p��w����@?w4˾����@��U��q-���x>�ӯ�c�c?vׄ�G��>������x���+>��D@�d�>�94��R�?7ʏ>(){?¦]����?�����W�>'I�>��@��y6�l���d?(��y>�K@��z�?����%�M���ؾ�}Y�*�M�g��=�7�?(��uyE�P6�Զ����?%�M�6ң=Q>]+�?&��>���>�g?uyE�TXi?�J�=�h?�G*�,H��)N<"�I���?�� ?u�_���t��C�~�h�W�?%�M�;j<�@�)�?����@��>ԽG?O?ڄ�������28��G*��q~���?"�I��x�Ӥ2?��?O?vׄ���1+��$���i?�q(��m��!��>hs@}% ��?�lf@z;�W�?��x��@=E���d|����?G�K��ͫ�uyE������i @������=j�n>A�_?�X?/�=�@;���>�?��Y�?�0�����?�G*������}Y�y+�e�A��%m�m�G�l��?^�N�`yD��$��٣�'I�>���)c��_���:�S?��c�c?l��>�w����@?����$��q(��Q]��%����\�?�'�c�c?l��>�~�?����O4�	�Q�����d|��+?E�V�7���-�cZ��������@?�G*�j�n>˝�a���ްc�Z���?�'���?TXi?�J�=�h?�G*�,H��)N<"�I���?�� ?u�_���t�k��?�o���h?)4?bN�ԡ�>�=�z>=֗�V�7�uyE�TXi?�q~�v
�?�٣�&�E���d|���C>�쮾�W?�>��ƿ_�?< ����[�(em�D���}>�?��P��?Ճ?�n@�{����?�9l���@Y;&?�q(��e���̽��x>�ك����>T6��_�/@ۓ�?w4˾�lk��}>�?C�ֿhs@�ӯ�(){?B����4����?�$��	�Q����=��ؽ�⛿�����7?l��?P6�l�R�b�����	�Q�D����=(������>��W�MF�����?`yD����?��>��+>���>���?���?=֗���g�:]��K@��3N�W�?-�@bN�Sux?��@�M�>�\6>��'?l�p=@��?�w��|��?�٣�T'F�E��a����d���@;��_>�j�?����?���<�|?f��A�_?
9�'�?�\6>m�G�:]�TXi?�q~��3?�G*�X�½g���w�>O%F?��*���w�:]�>2����������G*�,H���.?۲>��]�Ӥ2?F��?�>���7�?������=�S�r��d|��m�>�|����W��'���K@�l�R�W�?%�M��sX>��?Q��?�,���� ?��7?c�c?�$�>U�5���{>-�@����� <~<��?<�>��7?l�p=^�N��W?̢��K��	�Q�Q>Y ��N>ma?��?�?���R=��?���>w4˾�@=lk��a>�>���>������4	�B�����?�4E���=(��=D�����cq�?*�?���>�ξ2�"���?b�����>�h�lk��Q��?#��>*�?V�7�:]�����3N�QԾK���q~����A���k���� ?V�7���-�2�"��u?�28�w4˾�q~�D���*�M��q-�E�V�7�9Ӟ������ƀ>�9l��G*������R�?
9��1���+u?x�?��@�K@�Ln?���%�M�6ң=lk��aF�/
�<�>�ӯ���K?k��?z;��u?%�M��۠�E���f����?
]ڿw뿽:]����r�>�$�w4˾	�Q�������S?�d?ԽG?:]�)*>�3N��?w4˾L��@)N<R�>�u?��꿖�M@	���k���h�<�4E�K�򾂨�?���5����x>�쮾Gˣ?D����]�?	�ο�u?��:o��r�aF�&��>'0о �@���>��R=��G>Y;y��٣����>)N<A���B���=֗���W�O?)P=?l�R���@?K��,H��E����������4����?l�p=Fwо��96R���=�,;�D����=/�=��*����ξ@��?Զ���u?K��S�r�a���*�x��Ȱ�p׋��4	�2�"��ǿ]��>%�M�}�g���*��F���z�?�g?
��?2�"���+@�����W�>'I�>�}Y�J�l�eFC?�қ��ξ�$�>z;�JJ����x�;j<���>B�h�Ds�=3��=�ҋ?�?(.��Ln?����/y*��U5��}Y�5�����-���?m�G��n@������%?����K������}�?a>�>�,�������O?V��?����&��?�$��UX�>r��t�j.�>�����<���?��1��S4?�T���h~¾!���"�I�\�>:�S?�ӯ��ĝ�ʏ>�s����?�G*�bN����>y0F?@�1��� ?���>���?)*>����?�$��v����}Y�!���u?}j���қ�:]�TXi?�h�<+�?�$����=�F�?aC+>��C>G�K���'?�b�?2�"��J�=l����=f��D���*�t?��7��� ?f��>O?B�����?�F����=��r��U�O%F?�쮾hݏ��'��^�N��J�=����W�>�h��}Y��f� ��?��@ ��?��t��G?Զ���h?N�G��*����>�m��L�n���\��û?�?%Tl���?JJ��K��,;���@�R�>\�>�z�?���'���{�G��>�����O4���;���ŀ��Fo��3��=o���4	�P6�z;���?��=J��D����r���n?�FC?u�_�y>>2���h�<Y;y�K��h�0iG?
9���E���?��>L� �2�"� Ý>���?�٣��i?˝�a�����t>3��=��L� ��{� Ý>̢��w4˾&��@�VJ>>Z˿��x>��'?�b�?ʏ>�|?����z� ���;��@��U�>���m�G��ξ��K?��B?�>��=�.>D���}>�?�%?�쮾��<9Ӟ�cZ���񱿻��?�W�>���>����>>���P?�� ?��>�>���1�
��>��x�&�r�5���֛r�:�S?���>(){?��1���>�#���G*��$��}�?��ؽ�(�Ӥ2?��y>2�"��OQ?������x���ؾ���	�.�5u�Ӥ2?���>�?���=��l��K�򾟄$�Q>1䃾�jS�a�}��͛?�?l��>z;��>K�򾖍����@��e��?2=E=f��>l�p=TXi?a�?QԾ�!@��?��?�Ce?�f�?c���?�'���t�¦]���G>�!��z� �f��Q>ဲ�/�=:�S?���ξk��벪���?z� �&��q(�Y �/�= "?m�G�:]�¦]��S4?̢���٣�	�Q���@��t�L�n���x>u�_��4	��@/��>W�?'}�?�܍>E���d|��w@'0оx�?ά?��d>벪�懡?�O4���?D����m��g��='0о��:]��K@�1�
��>w4˾�R�r�B�h��V޿3��=�қ�ά?�Y�?1�
�N�&?��=(��=!���aC+>�!Y?-��w뿽�ξk��?벪���@?�O4�f�����>�f���.���X�¬�>ά?�����w��]��>w4˾J��0iG?۲>��7��\6>��Ͼ��t�P6�/��>�F���٣��$�A�_?"G�?s)���|���ك�MF���]�?K���@	nF@v�����@�Ce?�Q�?�%m��͛?���?��ƿ@�����٣��聿���=*�t?�V޿B�@¬�>c�c?�G?`yD�N�&?��=�۠����y+��Da?�\6>��?_3_��@ Ý>ۓ�?%�M����r��m����?
]ڿ(���ξ�C�~�h�W�?%�M�;j<�@�)�?����@��>ԽG?O?r��@�g?2�@��h?�܍>r���ؽ�[@�����>�R4?Fwо1�
�&D�=�٣�:o�����>ɓ?=Do>ma?w뿽:]���R=� �?96R�K���m=)N<qE ?l��G�K�Ճ?uyE�Fwо1�
�&D�=�٣�:o�����>ɓ?=Do>ma?w뿽:]���R=��G>���w4˾J������>>�&��>-���ͫ�uyE�¦]����?�����$�����>�M7�Y ������z�??�'�y>�C�3
۾���><t&@v���D���\�?'�?3��=�ҋ?l�p=TXi?�W?�$��$���h�r��Ѿ�M�>�������t��߰�r�>�7_���x�&��@A0>ˍ[��z�?��'?�R4?��?��o��>�O4�&�E��5������@;����=l�p=������?< ���G*��,;�r��Q]�4��>E����'��¦]���>b����_+@bN�g�Q]��,�� "?��Ͼ���>��1�/��>������v���)N<q
¾�z>��?7ʏ>MF��Fwо�3N��ޗ>K��v����}�?Q��?�+.> "?��	���)*>�W?�+� ��>q��>���>M�@�+?�����W��'���ۚ��ƀ>�7_��٣�͓\�����ى��,��:�S?��'?���>�{���?̢���$��&�lk���y6��`~���?�ӯ�O?cZ���w����?%�M��$�D����Q�>'�?�|���ԓ�	����Z?�ǿW�?��>;j<lk���ѾO%F?Ӥ2?u��?�ξ�����h�<�F����x�	�Q���?� �?�<�<�쮾�қ�uyE�������>�T��G*��q~����=�ii>�m�>�FC?��Ͼ�ξ%Tl��|?Y;y���r?!���@�J�l��y濇FC?��W��r�?ad@l�R�Q��?�٣���=r�	�.��E�?c���f��>��>���?��1��?��>�BB?����e��@=֗�7ʏ>_3_�@��?��o�&��? ��>�BB?����*��+?z[	���<l��?�x?������?��x��h�˝��t�cq�?�@;�ԽG?uyE���1���o��>�O4��$����?�6?�L�=:�S?��w���t��տa�?�9l�z� ��@=)N<y+��]��՛@(���b�?@��?=��+�?��-?h~¾���������C>����uyE�wh@I}'����?XD_>c�>���>�*��ó?��*���'?��K?l��>3
۾���>�$���$��.?*�t? ��?<�>u�_�:]���K?Ln?��{>%�M�;j<]?��?�u?E�u�_�9Ӟ���d>����G>%�M�i�X?0iG?9
?#��>�Ȱ�x�?uyE��K@�1�
��>w4˾�R�r�B�h��V޿3��=�қ�ά?ʏ>G��>QԾ��?�.>�F�?!�.?gd?-����w��ξ%Tl�I}'�n N?XD_>�q~�]?�ii>/
�Ӥ2?��I@�>����l�R�|��?��	<�?E���e�&��>�|��Q�@��t���d>I}'��3?L65@	�Q�A�_?�{��$C��s��7?�r�?����ƀ>̢����=�h�g��ဲ����@��>u�_����>�$�>�4�����>��=ۺ/����B�h�پ�ma?�_>���?lf@`yD�K�?�G*�6ң=r��Q]�U@�?a�}��ك�:]������-?����w4˾J����@�J�l���7�<�>��Ͼ�?���p��B�=b�����?���}�?�w�>l��@��>���=y>�����-?������@��E���Q]�wd|?ma??�'�uyE��]�?o,��h?w4˾�$�r�����?-����'?c�c?���e@�M��w4˾�.b�lk��2��>�(�@��>�ͫ���>�i�?xua�1��?����m=ԡ�> <~<��>�4����l�p=������G>QԾ�����>D��� <~<��7�G�K���?:]�^�N��W?̢��K��	�Q�Q>Y ��N>ma?��?�?�ʏ>���h?��=�����}Y��Q]��A��2=E=�W?^R@%Tl�=��l��e��@��ؾ0iG?�&E@�y��a�}�����K?%Tl���G>�9l� ��>�,;��}Y��>>��VԼ�+u?hݏ�_3_�Ţ.?��G>�ޗ>�$���@=���?�X?�z>��X�¬�>���?����z;��+��O4�J�����>���?lȾ�� ?p׋�:]��տ�7�?Y;y�ǂ��	�g�!���q��?��*�?�͛?,V@��,h���>��>ZW�lk��۲>�A�����>ԽG?���?�{� Ý>̢���$����;���>"�I���¿<�>��<�r�?Fwо�J�=�28�%�M��sX>0iG?ɓ?�?�� ?hݏ���t�ad@�W?1��?��[�,H��r�y+��?W�ɿ�ӯ��4	�P6�1�
����>�G*�:o�������;=&��>�\6>�қ���t�l��>lݿ�$��٣�	�Q�1Q�����Q�? "?�ȫ?L� ���R=`yD��7_�z� �:o��r�A�&��������*˳���-���1��3�?< ����@J��˝��d|�)�5?ma?��W�_3_��@xua�5@�G*�	�Q�D���	�.�B������}% �ʇ@����z;�������[��*�lk��y+��,��@��>У�:]�P6�0�?��{>�٣��,;�)N<
�U?B ½Z�����w��?�2�"�/��>���%�M��q~�Q>}>�?���>�˴??�'�:]�cZ����%?l��w4˾��+>E��bm?J�>=֗�u�_��ĝ�ad@��W�?K��v������aF�&��>I
���<MF��2�"�ؒ;?�!����	�Q��}Y�5�������\6>X�ӿ:]���?��G>�T��O4�	�Q����aF�X���4���ك�MF����>�����{>XD_>f�����>�@�Da?2=E=u�_���t�ڄ��*ߩ?���� ��>���Sux?aC+>K�ǿ*�?�ӯ�(){?2�"�3
۾d/�w4˾	�Q���������E���g�MF��(.����%?JJ�����S��M7�"����=ݰ�?��<�?�P6�o,����w4˾	�Q����>�d�>�ߧ��|��p׋�_3_��]�?I}'�v
�?��?�q~�]?�ii>-�?�4���û?MF��)*>r�>�9l�)4?��ؾ��@��e�4��>z[	��ͫ��ξ��Ln?W�?��?ԡ�>�w�?/�=��*�f��>�'������r�>���>��x�v������>ɓ?�M�>'0о(��D���¦]��S4?̢���٣�	�Q���@��t�L�n���x>u�_��4	�P6�1�
����>�G*�:o�������;=&��>�\6>�қ���t�)*>�3N�&D�=w4˾
��>���>	(@�z>��WA�?�P6�,h��]��>�����V���?���?�b�R�/��ͫ���@¦]��OQ?����K���$�]?B�h�>ma?x�?��-��Y�?�4�����>�O4�:o���q(����J$��s�ҋ?�R4?�C�3
۾���><t&@v���D���\�?'�?3��=�ҋ?l�p=��R=�w����@?�٣�����.?Q����%���w��ԽG?
��?ʏ>I}'�&��?��x���=�q(��{�wd|?��?��<l�p=��d>�|?�$�%�M��QB@E���t���?2=E=�ӯ�	�������ƀ>̢����=�h�g��ဲ����@��>u�_����>2�"���?b���%�M�	�Q����=���?#��>*�?m�G���-���p� Ý>�X��G*��q~��.?Ռ�?�Z�=��*���:]��C��3N��7_���=��ؾ����d|�پ��|���ԓ�l�p=cZ��/��>�+��G*����g��B�M?�n��@;�У��'���ۚ�o,�Y;y��S�?,H��g��IO�P��?���b�?�$�>��>��?�G*���+>E���>>�wd|?���>��L� ��߰�G��>Y;y��٣�,H��)N<	�.��ɿP��?��7?�?¦]�@&D�=���?h~¾lk��B�h�*g��˴?V�7��?B����J�=�$���-?�q~�)N<�w�?&��>Z���hݏ���t��x?벪�N�&? ��>j�n>���y+��@�7�? ��?��-�2�"�z;��T٣�,H��D���"��C�>�z�?���>uyE�TXi?�ǿ�u?z� �&�Sux?�{������\��ҋ?�>k��`yD�|��?w4˾J���M7��X?�z>�\6>?�'�D����x?l�R���?�O4����?�}�?E�?Т?�4��}% ����a�?�g?96R���>�h�r�"���Da?G�K�7ʏ>9Ӟ���1�1�
���@?��@h~¾��?A@ @j.�>Ӥ2?���>l�p=�i�?�w���h?w4˾J��r��t�?2����<D�����>����G>XD_>��=���>e�?j]??2=E=��:]����� Ý>̢��w4˾�۠��q(��Q]�/�=@��>��:]�V��?1�
����>)4?(��=0iG?�6?_O(?��*��_>l�p=^�N��W?�#���$���q~��M7�B�h��M��\6>m�G����>�x?K��W�?��>�h�g���U��n�z[	����b�?�ۚ���o��$���x���+>���>�{��	��*�?��Y1�?a�?�g?��q��$��J���.?�p;@'�?��X�?�'���t��?�w��W�?��?	�Q�ԡ�>*�t?��?G�K�7ʏ>�ξ�i�?��o�ۓ�?�$���G?˝�ŀ��&-��@;�m�G�ά?���>�U@�7_����>��E��5���B�����?�ӯ��R4?����6]�?�X���=���E���t�\�>���>(����K?����k��?��q�<�|?UX�>Չ�?�X?U����쮾�қ�D����K@�3
۾�>��>J��)N<y0F?�+?wm�?�ӯ�:]��C�֤$@< ���٣�D��ԡ�>�=�(>���?�'���-�Ţ.?���?�7_��٣��,;�r�"��wd|?z[	�(��9Ӟ��Z?l�@�>w4˾�,;����?��J@�+?�4��m�G�:]�I�?\������?�٣�f��E���t� ��?I
�7ʏ>D�����d>�W?�l.=��>xB>���=!�.?��>�쮾��_3_�������>�ޗ>w4˾v���0iG?�@/�='0оV�7���t���R=��>&D�=��=���>Չ�?M�@�`,?�4����w��ĝ������B�=�4E�%�M���ؾg��ဲ�lȾ�\6>hݏ��b�?�ۚ����?�����٣�bN�r������V�ݰ�?hݏ��ξ�xu@��#[?��x�:o�����	�.����?W�ɿ0Г?D������o���>z� ����ԡ�>1䃾�cy�E��ԓ����>�j(@�q~�W�?�!@ZW�E��ဲ��@�@;���<���?B���6]�?�7_���=�S�(�2@\�?�S?�7�?m�G��ξP6���?���>XD_>����?aC+>\�>���u�_����>�]�?	�οW�?��:o��r�*�M�]V?��\��!@O?�x?Ln?96R���[�f����@�	�.�����3��ԓ���-�k�����u?��x��i?�}Y�*�M����2=E=��l�p=cZ���q~��ޗ>�$���q~�g�� <~<v"˾Z����w?O?��p�,h��N�&?��&��}Y�*�M����X�7ʏ>�?)P=?�J�=��@?�$���S�A�_?aC+>\�>�@;�f��>c�c?��@�-?W�?r��?��+>����y6��\Y@<#�x�?�?�@ Ý>��?K��,H��E���Q]��+?������ξ�{��i @�����x�ZW�D���*�M�q�M��˴?(��MF���x?�q~��h?%�M�(��=ԡ�>}>�?l��?�4��*˳���t��Z?z;�W�?��=��c@E��*�M��Q�?����W?y>������%?< ��K��ZW��q(��e��@q��z�?ԽG?�>(.���OQ?Y;y��٣�(em����>��;=�cy�]��?�ك�c�c?��1�s�#[?%�M���ؾ���="��[�꿳�x>V�7�
��?�i�?xua��3?�G*�;j<r�J�l�?����g?�'��2�"�l�R���?XD_>��ؾ�.?�{�ڎ׿�\6>���>�n@�{���%?��G>�J0@X�½�}Y��==Do>��x>��>L� ��C�bԟ���?ǂ������}Y��ى����E���W�(){?vׄ��h�<b���w4˾}���?A����z�<�>�ҋ?L� ����?�3N�W�?��x�X�½E���t�� �?G�K�*˳�D����{�G��>�#��K���$���@�A�&�#��>��@¬�>l�p=2�"�`yD���{>�������R�?2��>`�Y>�+u?ԽG?MF��Fwоr�>JJ����x�,H���}Y�����7���X�?�'���-�V��?���?z� ��$����	�.�O%F?�@;��û?��t�Fwо�s����@?w4˾T'F�r�	�.��n�:�S?Ճ?ά?��p�벪�W�?��x�xB>���>�d�>`�Y>������=9Ӟ���ƿ�4��]��>��[�v���D����Q�?ɔ5�2=E=��D���TXi?z;��$��G*�c�>r��d|�'�?�\6>��W��ξ�����?�9l�XD_>f��Q>�=cq�?�FC?���>�?�a�?1�
���?��x�"�>��@���wd|?R�/���w�uyE��$�>��d>�>�W�>��=)N<�w�?���?<�>hݏ�:]�¦]��OQ?̢�����]Q�E���ى��,���7�?��W�:]�TXi?K���u?��x�6ң=����*��B���\��_>ά?�C���B?����w4˾f��r��y6��S?�R�?���?MF��B����4����?�$��	�Q����=��ؽ�⛿�����7?l��?����h�<b�����=��ԡ�>A���l��<�>��Ͼy>���l�R�|��?��=�۠�]?���?�z>2=E=7ʏ>uyE�I�?,h���?��?���>����y6�-�?��X�F��?9Ӟ����G��>��q�K��۠�0iG?}>�?�ߧ��|��m�G��?�)*>���>�W�>�q~��F�?m�+@&��>�|��m�G���t����=U�5�N�&?�!:@�]Q�˝�����"R���X�(�����?�$�>U�5���{>-�@����� <~<��?<�>��7?l�p=����`yD���@?w4˾���q(�aF�J�H��� ?��<�b�?��?��>�$�)4?��>ԡ�> <~<?�쮾���=uyE��Z?K����?%�M�}��q(�����8?�s�}% ����>�@�3N���?�W�>c�>���?���?��	@a�}�t�	@(){?��?z;���@?w4˾��;���>A0>O%F?a�}����>(){?�@��K�?w4˾,H��r��t�?!��ȫ?_3_���d>����>XD_>ZW��������7�3��=���=�R4?�Z?��B?��?��?X�½���>�d�>ꉒ?���>f��>(){?vׄ��OQ?����'}�?�*�g��*�M�|��d?(�����?>2�������������E����1S��@��>�ͫ�9Ӟ����W?Y;y�K���q~�lk���f���2=E=V�7�:]�����o,����K��S��M7�1䃾�Da?�FC?}% �	���%Tl� Ý>Y;y����sX>lk���?�����X�V�7�D���ڄ���ƀ>�l.=�٣�͓\�!����=>Z˿ "?�ӯ�ʇ@@��?�3N�K�? ��>������y+�/�=a�}�ԽG?l��?(.���h�<�9l�S"�?�G?�M7�"��/�=7ٛ?���=O?��1��C�?������,H�����> <~<NAc�E�У���t�cZ��	�ο�>�2�?&���@��>>��M�>-�����?��>2�"����?< �����?������ى��,�� "?X�ӿ��>k���3N�l���٣�,H���M7�A�&�������x>�����?vׄ��h�<b���w4˾}���?A����z�<�>�ҋ?L� ��?���&��?��x���?r�*�M�\�>�쮾��<Y1�?�x?������?��x��h�˝��t�cq�?�@;�ԽG?uyE�)P=?/��>1��?�G*��,;����?�k>?��?7ٛ?����-�)�$@�s��XM@��=��+>�q(���cq�?�@;���7?���>>�:@�0��p��?��=J��r�aF����?a�}����=��-�@��?�OQ?��?K���:�r��e�&��>�%m��ԓ��ξl��>U�5�1��?��,H���}Y�J�l�_�����*��_>l��?�K@���>����z� �T'F�˝�ŀ��lȾ "?�ԓ�L� ��]�?�o��|��?K��ZW�r�����D�>3��=�ȫ?��>cZ��=��W�?��=f�����?a>�>?�\6>��?0�̿2�"��h�<�28�w4˾��$�Q>�>>������*���>MF���@l�R��?w4˾ҧ<?�}Y�*�M�-�?�s��'?�4	��Z?�i @���%�M����>���?V�R@�w@ "?���=�R4?���� Ý>�28��٣�'I�>���=Q���V����s�m�G��R4?����B?����O4�����a����,��ma?�ԓ���t���>3
۾W�?��=f�����>�~]? ��?��x>����-���/��>������=�U5��}Y����KH=�<�>(�����>2�"�/��>���%�M��q~�Q>}>�?���>�˴??�'�:]��@xua�v
�?��=X�½A�_?Y'?H9�?�4��Ճ?�4	����/��>������[�}�g�Q]���#��d?(��E9�?��ƿz;��9l��G*���V����=���?�ο"�@��,V@�$�>��?QԾS"�?M��?BcW@�&E@wd|?�%m���?D����K@���?�T�N�G�&��}Y�A:�='D^�}j���ӯ��'���C�A�>�T$��	�Q�lk��2��>�B����V�7��?��G?�0���3?w4˾6ң=r�J�l��+?��X�7ʏ>��-�^�N�I}'���G>w4˾�,;����>A0>U�����??�'�:]�(.���;�?������x��������?aC+>�ȿ]��?��W��b�?�m�?I}'����?N�G���+>]?�*�4��>�O��'?MF��¦]�AH@b���w4˾bN�����ى��޾�˴?w뿽��>�������̢��z� �f��!���"��%K8�@��>V�7�l�p=�����D�>�F���٣��h�D����?��E���g�D���ʏ>U�5�N�&?��x��,;����>aC+>�+?:�S?w뿽D���Fwо=��Y;y���>f��lk����;=/�=��*���Ͼ�ξ%Tl�$��?�4E�w4˾��D���]+�?B���<�>��W�9Ӟ�I�E@C> ���@ ��>�R����=��ؽ�@0ҿ�g?�ξ�����W?|��?z� ��܍>�M7��Q�?�L��G�K�u�_�:]����=�w��N�&?���?�U5�˝�a���G�><�>��<�b�?Fwо�C�?�����٣���ؾ�F�?ဲ�
?��7�?}% �c�c?����6]�?�7_�)4?��m=��������u?@��>�ك�:]��x?��#[? ��>�������ŀ����:3��=u�_�E9�?l��>���?������=��=g����cq�?@��>Gˣ?��t��$�>�4��+�?w4˾��?Sux?Q��?��C>�쮾m�G���-�>2��U�5��-�����;g��A�&��(�Ӥ2?x�?9Ӟ���p�	�οN�&?�O4�ۺ/�g��A�&���2=E=�ȫ?0�̿������G>�4E�w4˾J���}Y���\�>���>��w���t���1��w���l.=w4˾J��Չ�?Y'?!&�=�d?p׋���t����?�q~�|��?���>��m=E��aF�wd|?�%m�m�G�l�p=3�?�OQ?v
�?�G*�f��r����� �?�%m�f��>:]�T6��G��>96R���ZW�Q>���p��wm�?��<��>��p��h�<�$�w4˾���lk��"G�?lȾZ���V�7���t���1���>̢���G*�J��0iG?Ռ�?l'�>wm�?m�G���-�ad@xua�2�@r��?ZW��}Y�1䃾U@�?-���ӯ�,V@�ۚ���&D�=�٣�v������?e�?��3��=u�_��?��@1�
�ۓ�?���>���>����y6�� �?�4���͛?�4	���p�xua��X����h��.?�=�n���X����=�'��wh@l�R�ۓ�?��=;j<����*��w@�%m���>�?��{���B?�����W�>?���>"��s�s��d?��L� ��a�?U�5�]��>K��S����=!����:'��}% ��R4?��ƿ��?�!��N�G��$�r����L�n�hs@u�_�9Ӟ�2�"���?�#���G*�������@��r� ��2=E=��g����?����z;������x�(��=!���y+�����+u?�ԓ��ξ��p��q~�JJ��K��۠���8@���?��7�Z�����:]�¦]���Y;y���x����Q>1䃾v�#��|��}% �uyE����=��Q��?��(��=���?Z��?g��=�|����?l�p=�{�o,������x���;��@��t�5u�ma?����t�@��?-ֿ]��>��x��*��M7�Q���s��z[	��ك��'��P6�l�R�#[?K��j�n>r��t�5u��|��w뿽c�c?�l3@��懡?��x�,H��D���ဲ��@�Ȱ�(��	����$�>(���h?��:o�����aF��N%� "?��'?ά?l��>�;�?������=xB>g��y+�cq�?��x>�ҋ?��t����0�?�M��W�@j�n>�F�?Ȱ?�?�@;���D���¦]��OQ?d/���w�+?!���*�M�ɔ5�Ӥ2?���=��K?(.��=���9l��G*��:��R�?9
?D�> "?��>�4	�2�"�l�R���G>w4˾ۺ/�lk��!��z房z[	���'?��K?�K@�/��>�>�*�?��ؾԡ�>��?Т?�|����M@_3_��K@�z;����w4˾h~¾˝��>>��n�'0о��D���%Tl���%?�X��٣��$�g��M�@%K8�@��>7ʏ>��>�C���>��q�w4˾�$����=�d�?�����X���W�MF���C� Ý>�>�$��v���lk��a>�>/�=�+u?��Ͼ_3_��K@�r�>�28���x�:o��lk��ဲ����@��>��Ͼ��-�¦]�G��>�M�����h�g���?��C>��?���'��k����o��>��v������=	�.��֬��s�7ʏ>c�c?�C����?������x��$�Q>����> "?�ӯ���t�cZ��`yD���?N�G���$����!���,���쮾x�?�?��ۚ�`yD�n N?w4˾,H���M7�	�.�i����d?�ȫ?�R4?)P=?Զ��懡?��x�v���!���A�&�U�_�@��>��'?
��?��?�0���$��٣����lk���Ѿ5u�-����<ʇ@ʏ>	�ο���?%�M�X�½A�_?�Q�>�S?�� ?V�7�D���2�"���G>��q�w4˾X�½!���.��?e��쮾��W�MF������1�
��K@K����A]?�X?�M�>=֗��y@��t��ۚ���G>�9l�z� �6�@�0iG?�᾽]����?�ӯ���K?I�?��>��?��=��!����=?��\�}% �O?��p��q~��u?h<U?	�Q�lk���m��j.�>��x>Gˣ?ʇ@@��?벪�W�?��=���E�����&��>
]ڿx�?l�p=�����J�=��q�w4˾;j<�.?��;=��8? "?x�?D���B����S4?�7_� ��>�@=���="�I�D/?E�?�'�:]�P6���%?�F���$���.>E���Ѿ�ߧ����>��O?��ƿz;��9l��G*���V����=���?�ο"�@��,V@� @�o��懡?��,H��r�����/�=�@;����>_3_�>2����G>������x��,;�r�a����y��ݰ�?V�7�(){?%Tl����$�z� ��h�Q>q
¾�y���� ?�ӯ��>�G?`yD�#[?�٣�UX�>�R�?y{@'�?c���s�a@:]�k��?;	@�>��=Rz?�}Y�2��>:	?!�\�Q@_3_���R=��o���{>��x��S����?�|?��7��|�����>O?B����s��W�?<�|?��m=g��	�.�L�n�Z���w뿽�j�?k��?U�5��3?��x�f���}Y��e��Da?}j�����=�'��)*>xua� �??@�$�˝�ŀ���B?Z�����O?Fwо�g?�X��G*���?ԡ�>���?��7���*�hݏ��'���߰�벪������A!侏�@Pu\@c�>3��=V�7�L� ����C�?�#���٣����Չ�?��ؽcͿ "?�����?�7@�w��懡?��?�.>���=�VJ>�@
]ڿ�w?:]��K@��"@���w4˾��D����X?P����x>�ӯ���t���ܿ�C�?QԾN�G�Er�?�}Y�R�>���j��?hݏ�O?�$�>� �?�T�%�M��h�lk����;=���>�@;��ԓ���-�l��>��o���?��bN����>a>�>֛r���\�Ճ?O?��1���_?�����G*������=y+���?�d?m�G�:]���d>�4���u?��[����g��A�&�����@��>��'?9Ӟ�ڄ��r�>d/����h�Sux?ဲ��	��2=E=Ճ?�b�?I�?I}'�懡?�W�>�$���?a>�>���?�쮾^Ƴ?O?�Y�?�S4?]��>�2�?
��>g���?�F@2=E=w뿽�4	�TXi?����Q��?w4˾�۠�˝�ŀ��ܷ=��4��¬�>c�c?cZ����+@�9l��G*������}Y��m��q=2=E=?�'�	���P6���?�ޗ>��>;j<)N<�m���z>�����y>%Tl��J�=�28�w4˾�q~�lk��Q���j.�>�R�?w뿽�?��C�U�5��>z� ��$���?�|?g��= "?}% �:]��K@�Ln?Y;y����sX>]?� �?����\6>V�7�D������ Ý>�M��w4˾�۠��R�?A����s��Ȱ�*˳��'��Fwо�4���>%�M�J��Sux?ဲ�ڎ׿-����Ͼl��?T�P@I}'�K�?<�|?�@=r� <~<�@p����w?���>�]�?Զ��K�?XD_>f��E���>>����?�쮾x�?�R4?������G>QԾ�����>D��� <~<��7�G�K���?:]�vׄ�����d/��W�>���D����f��'�3��=��Ͼʇ@����z;��9l���x�?����{���C>��\�u�_�:]�^�N�Y��?JJ���ϛ?v���)N<�?>�c?�˴?���>���Z?3
۾��?%�M��S��R�?ɓ?�T�>�4�����?�V��?�o����?K��	�Q�r�	�.�&��>��\���@��>Fwо��@�����٣��,�?E��a���0��7ٛ?��˿��>Fwо��o���G>z� ��,;�Sux?�?�jd>�+u?u�_�:]��@3
۾W�?��[�:o��˝��E��g��==֗����4	�(.�����?�!����?,H���}�?۲>��k��R�?�ك���@��ƿ��B?������x���˝��ى��ȿ�=%@��ʇ@��1�1�
��+���J�ｲ}�?A0>�gX�]��?w뿽�4	��]�?�D�>��?��=�q~�g��	�.�� �?a�}��_>	����ۚ���G>�9l�z� �6�@�0iG?�᾽]����?�ӯ���K?�������l.=�٣����.?�ii>�+?Ӥ2?x�?D���)P=?������?��[�ۺ/��q(�	�.����=-��Ճ?L� �2�"����?< �����?������ى��,�� "?X�ӿ��>l��>���?�F����x��,;�D�����;=&��>��\���g�uyE��Z?z;��>�G*��.>���
9���?�s�ӯ���-�ʏ>Զ���3?<t&@�۠��}Y����em���|��u�_��j�?�?xua���?��?,H�����=�?q�?-��ԽG?9Ӟ���1���?b���#Xo�h~¾E��5��� )���%m�p׋��?�(.���ƀ>�4E��G*��7?A�_?Q����R���?f��>��K?)*>B��?96R��G*�f��D���"���u?ma?p׋�MF��2�"��W?����K���sX>�M7�1䃾�.�;���>��	���Fwо�-?�!��K�����M7��Q�>g��=���>�қ�L� �>2�� Ý>�X�%�M�F s�)N<�6?6Eڿ�+u?}% �l��?Ţ.?��W�?�!:@���E���e�?��*�¬�>O?)P=?��_?�$���x���E����?c����ӯ�_3_�¦]���o���{>��!����ŀ��Ӌ�@��>?�'�ʇ@���G��>�M��z� ��q~���?2��>�s��Ȱ���˿�'���C��OQ?̢����x����Չ�?Ȱ?"R�2=E=��>(�?�x?�q~���?w4˾D��r��ى�x�z[	���W����?%Tl�eU�?���w4˾�܍>ԡ�>�Q�>��;��?x�?Y1�?��p�z;��ޗ>w4˾f��lk���p;@/�=�@;�?�'�:]�ʏ>�4��N�&?XD_>���>�}Y�J�l�s������W?���?P6�3
۾�X���x��.>)N<!���,����*��ك�_3_�Fwоa�?����K��܍>r�5���/YO�@��>��ϾO?^�N�Y��?JJ���ϛ?v���)N<�?>�c?�˴?���>���P6��ǿn N?z� �!�]?A:�=�	��<�>�ӯ�ά?)P=?��G>�X���-?'I�>�@Ȱ?-�?�|��}% �9Ӟ��������?Y;y��$��(��=����*��M�>�s�f��>�?�Fwо=�����S"�?���>���?E�?�R?�4�����ĝ�TXi?z;�1��?K�
Ao�$@��D@%��?��@@a�}���?�/��@��?벪� �?�$��:o��r��e�P��-�������?2�"�o,������[�v���E��*�M�j.�>2=E=�w?uyE�)*>T��?����XD_>��+>���=��ؽ��H?3��=�ӯ��4	��@l�R�W�?��>j�n>D����{�U@�?a�}�hݏ�L� �P6��ƀ>�X���?�۠����=M�@�.�;�쮾���'��%Tl���X?����$���@=���=%��?B�����x>V�7�9Ӟ�2�"��ƀ>b����O4��q~��q(��f��z>@��>7ʏ>uyE�1��=��������F s�Q>}>�?�?/�B�@¬�>S�:@>��������O4��:����>�)�?�����?u�_�E9�?��1�1�
���@?��@,H�����?O@j.�>Ӥ2?f��>l�p=����W?l��w4˾X�½���=��?`���쮾���?�k���W?����$����m=g���m���u?�FC?��'?��-������|?b���K��,;�ԡ�>ဲ�*�x�ma?�_>��>k�� Ý>����w4˾����}Y��ى�پ��d?�ﾞ�K?��>���?�$�z� �	�Q����aF�Ds�=E����=�ξ�$�>����$��O�>�.>���?�)�?O%F?�Ȱ�m�G�D����@��G>W�?�$�����E���f��X�?}j���ȫ?��>�x?��G>�$���x��,;����>۲>��?��\�(��9Ӟ�����z;��>K��:o����@��>>�������(���4	�����G��>&D�=�O4��i?�q(�!���M�>�쮾�Ǵ@�ĝ���1�/��>�#���G*��,;�0iG?�m��p�X�ma?hݏ���>Ţ.?eU�?��q�K��,;�!������N>'0оu�_��ξ�x?1�
��ޗ>��>��+>���?� �?�Da?-����>�ξ�K@��J�=�9l�w4˾�S����J�l�Y'���FC?��Ͼ(){?�����?�����O4�(��=]?"��(�R�<�>���ξ��1��OQ?�����G*�X�½Q>�f�@�>:�S?V�7���t���d>�|?�$�%�M��QB@E���t���?2=E=�ӯ�	���2�"�ؒ;?�!����	�Q��}Y�5�������\6>X�ӿ:]��C�o,���{>�٣�h~¾BcW@�6?N����\6>Ճ?�n@a�?z;��>h<U?J��E���d|� ��?E���W�y>�G?U�5�W�?%�M��R���@�9@'�?-��?�'���t�ʏ>/��>&D�=��x�j�n>E���t�?@��>�ك��ĝ��a�?�q~���?��=h~¾E��J�l��z>�4��}% �O?���l�R��#���O4�͓\�Sux?2��>��G�!@��?^R@��1�1�
���@?��@,H�����?O@j.�>Ӥ2?f��>l�p=���?=����?�g�=�R��q(�q
¾�t�>�s��7?�j�?����k��?������x�f�����"��B ½'0о��w��?��]�?l�R��u?�٣����>r�������V�'��}% ����?k��xua�n N?��=J��Q>%��?9>3��=��>��-�%Tl����$�z� ��h�Q>q
¾�y���� ?�ӯ��>(.��l�R�1+���x�~Q��lk��A�&�p�X�:�S?}% ��?)�$@��?ۓ�?�!@X�½�}Y��{�8'9@W�ɿ¬�>�ξ%Tl�*ߩ?̢����	�Q�r�5���lȾwm�?����徖�1���1+�XD_>�q~�lk���m���S��@��>�ك�y>��1�
������=�۠���?qE ?�������>��l�p=�K@� Ý>�4E�K��6ң=�R�?G�@5u�:�S?p׋�l�p=)�$@�s��XM@��=��+>�q(���cq�?�@;���7?���>vׄ���?����K�򾖍���q(��Q]��z>�+u?}% �D���B����?�M��N�G��$��}�?M�@�z>a�}���9@�ĝ�cZ��z;����> ��>�h����>�]@&��>z[	�u�_���t��$�>U�5��?��=T'F�!����f���k�2=E=u��?��.@2�"�;9�>̢��K��xB>���"����k��+u?m�G����?�����OQ?v
�?��Y;&?���?��7@/�=�%m�u�_���t���`yD��4E�z� ��sX>���>�f��EӾa�}��ك���t��Z?��B?��?��?X�½���>�d�>ꉒ?���>f��>(){?��1�����l.=XD_>��ؾr�J�l���/=�s�m�G�L� �^�N��ƀ>����XD_>�S����>�*��jS� "?��W����>k���3N�l���٣�,H���M7�A�&�������x>�����?^�N�I}'��ޗ>��x��S�]?Q����cy� "?��?�?�$�>���h?z� �!��}Y�5����ؿ'0о?�'���?��?�W?JJ����>�sX>��?�??��X���uyE�>��w���������]Q��}�?qE ?1;.�B�@�W?^R@2�"��7�?����w4˾v�?����ى�=����� ?��W��ξ�]�?1�
�1��?�G*���ؾ!����U�lȾ�Ȱ��ȫ?c�c?�� Ý>�#����;j<D����Q]�Ǜ�2=E=(��l�p=l��>��_?96R�XD_>��=g��A����.�?���?�'�:]�^�N�a�?�!���G*��:��}Y�J�l���`��d?��(){?¦]���?Y;y�K����ؾ���?1䃾j_��Ӥ2?Ճ?:]�@��?K����?�ϛ?(��=g��	�.����>E���@���?�Z?�w����?��[�ZW�˝�)c���A��G�K�?�'�c�c?��>3
۾W�?��=f�����>�~]? ��?��x>����-���/��>b���w4˾�,;����>�*�l��=�� ?u�_�9Ӟ���R=� �?96R�K���m=)N<qE ?l��G�K�Ճ?uyE��C��OQ?��q�����@)N<A0>g��=a�}�E�A@��t���ƿ;9�>< ���$���۠����>A�&�O�P��??�'��n@��1�1�
���@?��@h~¾��?A@ @j.�>Ӥ2?���>l�p=^�N�3
۾96R��$��d�x�r�5����ؿ�|��}% �c�c?¦]���%?������>�G?lk���Ѿ%K8�:�S?����K?��p�U�5�&D�=��	<�?1Q��A���j.�>�|�����?:]��{�G��>�#��K���$���@�A�&�#��>��@¬�>l�p=����߹�@������J��˝�5�����7���?p׋���>�{� Ý>������x��܍>�AQ@qE ?(0�ݰ�?��>c�c?�ۚ�1�
��F����	�Q����>�Ǔ>��]��z�?�ӯ�:]��C�K���M���O4���V�0iG?9
?c�ۿ<�>��Y1�?���?/��>W�?��>�R�����*�cq�?�4����l�p=�C�Զ����?�٣��h�E���ى�B���3��=m�G�L� ��{�˹�?̢����>������=q
¾�(�`�?�û?��?TXi?U�5���?�M@�sX>r��U�wd|?Z���Ճ?Y1�?I�?=��|��?��=�h���@�*�M��q?���m�G�O?�x?I}'��h?L65@,H��r��d|��x>��*��ك����?cZ��a�?�9l�w4˾!�E������N�����X������?�!@/��> �?�G*�}�r�A�&���?�����MF���Y�?3
۾�X�K��J������Q]��?-��}% ��'��ڄ�������28��G*��q~���?"�I��x�Ӥ2?��?O?TXi?�q~���?N�G���$�Չ�? <~<��6<�쮾Ճ?�>����o,����K��S��M7�1䃾�Da?�FC?}% �	������J?�����$����m=��@��U��M�>Z���¬�>�'���w�/��>�+��G*���$����=�)�?��|s_@�ӯ��n@¦]�Ln?����-�@,yM?���>۲>��?�+u?���>_3_��ۚ���o��$���x���+>���>�{��	��*�?��Y1�?����l�R��l.=�٣�ZW�!���Y �/�=z[	�¬�>��-�@��?�OQ?��?K���:�r��e�&��>�%m��ԓ��ξڄ��`yD�Y;y�z� �ZW�˝�*�M�ɔ5��\6>����t�B����"3@̢����-?f��r��ى��?�7�?�ك�O?��1���_?������x��۠�D����>>�g��=�7�?��w��'��Ţ.?���$���>(��=�M7�A����+?'��x�?�ξP6�,h��]��>�����V���?���?�b�R�/��ͫ���@l��>z;��>K�򾖍����@��e��?2=E=f��>l�p=�?��W�?XD_>;j<g�ᾬz>-����>Y1�?1�뿖����T٣��6~�)N<���?���+u?�ȫ?�n@��ƿ�J�=���z� ���V�E����� )���e)@��c�c?2�"��W?����K���sX>�M7�1䃾�.�;���>��	�����p��s����@]B�@�]Q�Ǟ@/��@n��p�����g���?¦]�G��>���N�G��*�E������x�I�Ӥ2?�ӯ�:]������>w4˾&�Q>q
¾]�P���X�u��?��K?�K@��g?����K��!侒�@qE ?�u?��?�w?MF��k��r�>QԾ��J��lk��q
¾��̽]��?��>l�p=�C�q@E@�9l�z� ���?���� �?�,���|����	������=����G>�O4��$�lk���{�'�?���w뿽	������=���$�w4˾bN����>A:�=?�*��e)@x�?^R@�߰��-?�4E�w4˾:o�����?A����,�*�?f��>_3_�cZ��6]�?������x��@=!���y+�'�?���>�қ�uyE����o,��-�K���R�D���ဲ�/�=<�>u�_�:]�2�"�Զ���ޗ>z� �D��˝����lȾz[	�.�1@�ξ�j(@z;�v
�?XD_>����}Y�	�.��C�?�����>��>Fwоr�>JJ����x�,H���}Y�����7���X�?�'���-�2�"��W?����K���sX>�M7�1䃾�.�;���>��	����$�>G��>W�?w4˾�@�.?��?�z>'�� �@�'���{��W?����'}�?��=BcW@��p?�+��@;�У�:]�)P=?l�R���G>%�M�xB>)N<�=O%F?G�K�7ʏ>�ξ%Tl��a:@Y;y��G*�!侱��>���u��z�?��O?FwоI}'�W�?w4˾;j<)N<��ؽ�+?�7�?w뿽�ĝ��i�?�w���h?w4˾J��r��t�?2����<D���P6� Ý>�l.=��?ZW����>S@j.�>-��V�7���t�l��>U�5�+�?G�?�$�r�*�M�������X���l��?��p���B?������[�xB>r�����@��FC?��:]������h�<�#���O4��h�˝��U���k�a�}��қ��?�2�"��u?�28�w4˾�q~�D���*�M��q-�E�V�7�9Ӟ�B����o����@?w4˾�$�g�f��M�>��*��g?��t��C�1�
��$�%�M�	�Q��}�?�X?�u���?��Ͼ�?¦]���_?JJ���٣�h~¾�q(��y6��q-�<�>�ӯ�L� �@��?��o�&��? ��>�BB?����*��+?z[	���<l��?vׄ�U�5��7_�w4˾�q~��F�? <~<H�"�@��>p׋���t�P6�o,��!��z� �h~¾���ဲ����\6>���ξ2�"���B?�!���G*�	�Q�)N<�y�?q�>�˴?����-���p�/��>b����O4���;lk���d�>��7��|���қ���t�����k��?����)4?xB>Sux?�w�>~�?��?���>�?���p��?�#��K���q~��}Y���U����� ?V�7�uyE������Z@�9l�N�G�lG�?�}Y��U�긹�7ٛ?p׋��?�k����?�7_��$���$�)N<�Ѿ�t�>`�?�_>���>¦]���B?����O4�h~¾)N<�w�>/
�E��ӯ��'��ڄ����_?�#����h?c�>���?�ii>e�*�?��<��K?l��>��W�?��=�.>��@�)�?O%F?}j����Ͼ�ĝ�ʏ>r�>�����x��q1?�M7��Ǔ>�,����*��ӯ���>vׄ���1+��$���i?�q(��m��!��>hs@}% ��?�w?�@��o� �?�)u@xB>]?�k>?&��>I
�0Г?j��@�K@�� �?�4E����i?lk��9
?e��-����Ͼ:]��K@�6]�?Y;y�w4˾��=����U�5u�ma?�ԓ��ξ�����D�>�F���٣��h�D����?��E���g�D������=K��1��?K��]�@lk��1䃾�gX�p���.�1@��t��?K����@�ϛ?�i?r�	�.�\�>0ҿ��<��.@���?K����@?��x�����M7��U�n �>'0о���4	���d>�J�=��q�%�M��sX>�}Y�"��&��>Z����ӯ��'������6]�?�X���=���E���t�\�>���>(����K?�$�>lݿ|��?���h���?2��>/�=�� ?m�G���t����C�?�#���٣����Չ�?��ؽcͿ "?�����?�� Ý>�7_���'I�>���?���?P��}j���ك��?�ʏ>��B?��q�z� �v�������y6���C>�4��x�?D��������h�<���z� ��q~����="���7�3��=(���ξ)*>\���N�&?��x��,;�˝��E��ˍ[�Z������=ά?�@U�5� �?8,����ؾ�}Y��f�?0ҿ��>��K?^�N������X�w4˾���!����>>�����ma?0Г?�R4?�$�>�3�?�28�H�7?;j<����d|�cq�?:�S?��W�(){?��p��s����@]B�@�]Q�Ǟ@��@n��p�����g���?�Z?�s����?z� �&�˝�a����y��c�����(){?�$�>�����?��?��>˝�����e���*���Ͼ�j�?cZ��Զ����{>XD_>!����5����8���|����'?ʇ@%Tl��J�=�28�w4˾�q~�lk��Q���j.�>�R�?w뿽�?����?,h��|��?N�G���ؾ�}Y��t�j.�>-��w뿽9Ӟ���1��4���>��x�6�@���?�]@l��2��-��@:]�P6�=���+�w4˾�۠�r��Q]��z>E�hݏ�D���k���3N�QԾ�W�>�h�)N<�m���������>��g�l�p=2�"�o,������[�v���E��*�M�j.�>2=E=�w?uyE�(.��`yD��ޗ>)4?	�g�D����VJ>�� �@��>u�_�E9�?TXi?��>�����-?'I�>���?�k>?�L�?���f��>�?���1����M��K��h����?�ii>5u����>w뿽�4	�k����%?������܍>��@��y6��L���|����uyE�>2���C�?�7_���=6�@�!����f�Q}ӿ*�?��w�ά?)P=?I}'�]��>�G*�6ң=����y6�:	?�s�(��l�p=��1���>�����٣�6ң=�q(��d|�|�Ӥ2??�'�(){?������?���N�G��q~���@��t���`��쮾m�G�uyE���p�,h��N�&?��x���V���@�X?lȾ3��=��?��K?�x?"�?n N?�G*����lk��"�𾈺���4����O?��ܿ��96R���x�͓\�)N<� �?��&���?^Ƴ?S�:@�x?��G>�$���x��,;����>۲>��?��\�(��9Ӟ���K?�h�<���?w4˾�E���U�x�|��Gˣ?
��?�@�ƀ>1��?��=J��r����@'��ԽG?y>Fwо��?��{>��x�J���}Y��ii>����@��>��W���t��������?�F����x����g���*���@��>?�'��ĝ��K@���G>�28��O4��h�A�_?�{�s�s��\6>��ϾD�����R=��?�4E��������@�r�:gf�<�>�û?���?�m�?=����?)4?���>��?e�?�w@�@;�/��?D������?�?ۓ�?L65@;j<r�	�.�?�w����@�)@cZ��	�ο�>�2�?&���@��>>��M�>-�����?��>�C�r�>�!���O4�������=�{�5u����>p׋�	����{�Y��?�!���G*�������?��;=]�P��z�?(����?��1�
������=�۠���?qE ?�������>��l�p=)P=?/��>1��?�G*��,;����?�k>?��?7ٛ?����-���ƿ��?�����O4��@=)N<�d�>;a��`�?��W�MF��cZ��`yD�W�?�G*�q��>0�>@	(@�u?<�>��@�?�2�"���o��u?�٣�v���0iG?A���ܷ=�ma?�͛?O?�G?����+�?w4˾v����}�?�?g��=��X��W?y>��p�o,���q�K��,;��5 @g�5@��7�z[	�V�7�:]��?����懡?��Rz?���=A�����>��X���7?(){?l��>*ߩ?��@z� �-��@Q>9
?�����+��Ն@�ĝ�B����o����@?w4˾�$�g�f��M�>��*��g?��t�cZ���o���>%�M��۠�)N< <~<@�ؾma?ԽG?��K?�C������T٣�h~¾D�����@�>�z�?¬�>:]��m�?����N�&?��[�6ң=r���8R#>W�ɿ}% ��N��l��>��|��?��!�Sux?�r���2=E=��@�>��I}'�2�@���x7A��@�
9�/�=0ҿ��q@��t�B���_�?1+�%�M���=!����ii>D�>@��>V�7�9Ӟ���?�?�?�T�x��?���Z��?g��=,8�"Ԍ@_3_��{��ƀ>̢���٣�?r�J�l��׽<�>�_>:]���R=�S4?���?�٣��3
?D���E�?j.�>�@;���W�D����߰�K�������O4��6~�˝��ى�V����7�?�͛?Y1�?k��r�>QԾ��J��lk��q
¾��̽]��?��>l�p=��>1�
��>%�M��$�Sux?�y�? ��?<�>u�_�:]��K@��ƀ>̢��w4˾������=y+���@��>hݏ�_3_��� Ý>�#����;j<D����Q]�Ǜ�2=E=(��l�p=����3N��M����=7p�@��@��d|�/�=��*�u��?l�p=����K���l.=��>�܍>Q>9o�?����-�����=��t��@l�R�W�?��>j�n>D����{�U@�?a�}�hݏ�L� ��a�?o,���?��=ZW���?�Ǔ>*g��4���W?�j�?�m�?I}'����?N�G���+>]?�*�4��>�O��'?MF���G?���?l��XD_>	�Q�0iG?�Ǔ>�Da?������=�>�@U�5��?w4˾���r�	�.�cq�?�⿗ӯ��'����K?�������?%�M��.>���?�ҍ?�Da?��X�Ճ?uyE����=ؒ;?W�?�J0@�BB?�R�?��J@�w@���E�A@�'��TXi?��>�����-?'I�>���?�k>?�L�?���f��>�?��C������{>�٣�&�Չ�?�ii>���>�|��}% ��ξ�K@���?�7_���[�,H���}�?qE ?lȾ��X�hݏ�D������?�񱿡�?z� ��:����=�ᾮZ�=�4�����=�>���OQ?�X��٣�;j<���=�Q�?�,��2=E=p׋�:]���>K��W�?�$��	�Q�Sux?��@'�?z[	���D���ʏ>G��>1+��$��!�����{�g��=G�K���Ͼ:]�%Tl����̢����x��$���?z��?�GK��Ȱ�m�G���-��������?懡?w4˾�BB?�@��N@g��=��*��_>�'������NY�?&D�='}�?X�½���>9
?&-�a�}�m�G���t���d>Ln?��G>�W�>���.?��?���?�7�?x�?�b�?B����J�=�l.=K��h�g��"�I�'�?�� ?hݏ��ξ^�N���o���བ$��j�n>E��J�l���7� "?�ӯ�9Ӟ�ʏ>I}'�&��?��x���=�q(��{�wd|?��?��<l�p=(.��U�5��X��O4�ZW�E��5���B����\6>(��D���Ţ.?��>#[?�G*�,H�����>Y'?��?�+u?��:]�Ţ.?�ƀ>l���G*�J���.?/b=@��?Z���?�'���t�vׄ�o,�96R����۠�!���1䃾�(��\6>��W��'��ڄ����������[��U5��}�?�Ѿ����d?p׋���?�x? Ý>�F��K���$��q(����M�>=֗�?�'�:]�k�����?QԾ�@�.>���>e�?�$T>a�}���W�uyE��@��W�?�*�?h~¾Sux?�=��8?
]ڿ��uyE�)P=?�i @�l.=�$�����F�?�5C@�u?�s�m�G���t���ƿ�4��O%@" Av����.?Ȱ?1׆<�� ?p׋�l�p=��d>o,��9l�%�M�6ң=ԡ�>�=&��>�\6>Gˣ?�>�ۚ� Ý>�����O4�������@��Q]�����P��?0Г?��t�����1�
���{>XD_>�$���?۲>�M�>���w뿽L� �ڄ���"3@����w4˾X�½���=	�.����ݰ�??�'��R4?(.����G>d/���	�Q����=�?�� �3��=�ӯ�l�p=����-@d/��٣�'I�>g�ᾬ���z[	�p׋��ξ��	W@n N?��x��i?]?S@\�>'0о¬�>D���)*>Զ��N�&?��=X�½)N<Y'?��?�|��p׋���t�B����S4?�7_� ��>�@=���="�I�D/?E�?�'�:]��e�?�o����@���<�R����A�&�\�>��\�hݏ����?TXi?�ƀ>���w4˾���>D���� �?�n���u��?9Ӟ�cZ���q~��ޗ>�$���q~�g�� <~<v"˾Z����w?O?%Tl����$�z� ��h�Q>q
¾�y���� ?�ӯ��>���e@�M��w4˾�.b�lk��2��>�(�@��>�ͫ���>�x?�3N��M�����@,H��!���A:�=�k!@a�}��_>D����@U�5�懡?w4˾(��=lk���ii>� �?�����<_3_��a�?�OQ?�F����>UX�>���=q
¾U@�?��X���@MF���G?���&D�=�٣��J^?r�J�l�/�=-�����>�R4?��p�&)��@?�O4��*�Q>Y �긹�E��_>�R4?^�N��3N�QԾz� ��,;�!���y+��,�<�>w뿽�?�cZ�������$�w4˾�,;�)N<�y�?�f�>�\6>w뿽:]��ۚ��S4?b����O4��]Q�r��Q]���ܿ*�?V�7�O��?���������@z� ����@�M7���B ½����!@�ĝ�������G>QԾ�٣��S�lk��"G�?�,��3��=V�7�:]���R=xua���?���2�@�q(�y+��u?0ҿ�!@�ĝ�� @l�R�v
�?�٣�,H��E���U�wd|?�w��u�_����>�m�?����N�&?��[�6ң=r���8R#>W�ɿ}% ��N������g?���%�M���$�E��a���NAc�<�>?�'���>�$�>(���h?��:o�����aF��N%� "?��'?ά?�K@�z;�1+���	�Q�����y6�q�M��˴?hݏ��4	������>�F����J��W&@2��>/m��Ȱ��Ȼ�D����{����?̢���k�@�BB?�Ο@G�@cq�?��?m�G�MF��>2����%?̢���$����;Sux?�r��3ֿ��?У�O?��� Ý>�!���ϛ?�$��}�?�r��޾�O��ԓ�D�����?�J�=��G>�G*��3
?g��1䃾�2>�s�(����>Fwо`yD���G>�G*�J���.?�k>?O%F?�z�?}% ���-�ʏ>��>�T� ��>�@=D���
�U?G�>c���V�7���t�Ţ.?xua�懡?w4˾J��r�y+��+?��\���g�:]�k��?����#[?�O4�}�)N<�m����n?-��ԽG?�ξvׄ�z9@�����G*�v�?r�5���ѩ&�P��?��g����>�$�>�����?��?��>˝�����e���*���Ͼ�j�?cZ��"�?JJ��z� �v���E�����/�=�@;���W�:]��ۚ��|?d/��2�?�q~�!����f�;�����?�ӯ�ά?)*>=����?����ؾr�5���!V���x>�_>Y1�?>2������9l�z� �	�Q�Q>Y ��]���d?V�7��R4?Fwо�W?�!��w4˾�$��}�?��ؽ��:�S?w뿽��>B��� Ý>�28� ��>xB>�M7�qE ?�Da?�%m�u�_��ĝ�%Tl����Y;y��٣����Q>q
¾�N%�3��=w뿽��-���d>=���h?�$��������?�d�>P��3��=��7?���?ʏ>/��>&D�=��x�j�n>E���t�?@��>�ك��ĝ�k����?�7_��$���$�)N<�Ѿ�t�>`�?�_>���>�G?��o���?N�G��$�@a>�>������?¬�>��>�?�"3@n N? ��>�q~�Sux?}>�?�Da?}j����9Ӟ�Fwо��o���G>z� ��,;�Sux?�?�jd>�+u?u�_�:]���K?�4����?�O4�&�!����Ѿ�S?Z�����<�ξ(.����o���?��x���G@���>ဲ�Hh⿦��>��?
��?�G?G��>��q���-?��m=A�_?\�?-�?�s�u�_���-��K@��J�=�����٣��,;��@a>�>5u��� ?0Г?c�c?P6�T��?l��w4˾�$�g��qE ?����E�p׋��?�¦]�Ln?����-�@,yM?���>۲>��?�+u?���>_3_���>��?���K���f�?�}�?E�?�Da?a�}�7ʏ>��t��$�>������G>w4˾;j<D����U�k���%m�?�'��4	��x?�o�����?K��X�½�5 @
�U?�#��I
������?(.����%?JJ�����S��M7�"����=ݰ�?��<�?��K@��ƀ>����w4˾��@D���
9���6<����ӯ���t�B���/��>�28���=�R�r�	�.��Da?�����ϾD���k���q~��ޗ>�$����ؾ�q(�y+��A��@��>�ӯ�ά?%Tl����$�z� ��h�Q>q
¾�y���� ?�ӯ��>)*>���?�$���H@Y;&?Չ�?;�)@�@2=E=��%@�'��B���G��>�>�٣�f��g��A:�=�z>ma?hݏ�_3_���?�h�<�$���=	�Q����?5�3@��?��*���:]���p��S4?�#��%�M�	�Q����=1䃾�n��\6>x�?Y1�?����/��>��q�w4˾�$�Չ�?�"$@;�����*���W�MF��ʏ>�g?��q��O4�;j<)N<�y6�!&�=2=E=�ԓ���t��m�?��#[?�O4�r4�?��?A0>,�[?<�>Gˣ?MF����ܿ��96R���x�͓\�)N<� �?��&���?^Ƴ?��8@�ۚ����+��O4��۠��}Y��ى�.Ũ��R�?V�7�c�c?�K@�/��>�4E���x�X�½!����������z�?u�_�_3_�^�N�a�?�!���G*��:��}Y�J�l���`��d?��(){?ʏ>3
۾n N?��[��q~�D���aF�ѩ&���*���?_3_�k���ǿ&D�=��x��:����>�=�C+�ma?��'?��>TXi?�J�=�h?�G*�,H��)N<"�I���?�� ?u�_���t�2�"��3N��$���x�h~¾E��Q���lȾ3��=p׋���t�k��?3
۾�ޗ>%�M�q��>ԡ�>1䃾�nC?�s�f��>���>�$�>Y��?�>��[��]Q��������H�"�'��p׋��ĝ�l��>�ǿ�>w4˾,H��(�2@�X?/� ?:�S?Gˣ?_3_����l�R�+�?�G*�"�>E��5���Pܺ�@��>?�'�c�c?��?1�
�懡?�G*�X�½�M7��y6�&��>��x>hݏ���K?Fwо6]�?����z� ��q~���?
9��ߧ����u�_�y>Fwо�����$��$��6ң=��@�1䃾�+?��?��<uyE��Z?`yD�n N?/y*���ؾE���ى��Ԅ�G�K��қ�c�c?cZ�� ��n N?%�M��q~�Չ�?A0>�J$��˴?�w?�4	�2�"��w����{>%�M���m=W&@!-@����2=E=�W?�?�K@���%?̢���٣�!��q(��y6�\�>��?���=��t�I�?�J�=�@�S�?,yM?r��t��Da?p����ҋ?���?2�"�I}'��>�٣�X�½���?y0F?\�>:�S?��W���t�k��a�?���z� ���lk����;=�i�'��p׋��ξ��Ln?< ����>:o��lk���f��M�>�쮾hݏ��ĝ�¦]�0�?������ZW��R�?aC+>&��>j��?��>MF��¦]���_?�9l�S"�?��ؾlk��!���GK� "?��W��?k����?< ��K����ؾ0iG?�{��L��Ӥ2?��(){?k����n@����͑�?�sX>E���d|�����"�@V�7�y>��1���1+�XD_>�q~�lk���m���S��@��>�ك�y>l��>o,��ޗ>#Xo�h~¾�q(�Y �'�?�s�ӯ��'��%Tl���%?�X��٣��$�g��M�@%K8�@��>7ʏ>��>¦]�T��?�9l���v������A�&�ܷ=�<�>У�uyE���R=��G>���w4˾J������>>�&��>-���ͫ�uyE�Fwоo,��4E��W�>!�D����VJ>��C>@��>���ĝ�2�"��u?�28�w4˾�q~�D���*�M��q-�E�V�7�9Ӟ���p��J�=�>%�M��q~���@��e�����@;����>]=@cZ���w��]��>K��*�W&@
�U?3L��z[	��w?���?��d>Y��?������[�����0iG?�Ѿɔ5�Z���?�'���-���>���X�XD_>(��=���?�d�?�S?�Ȱ�(��D�����d>Ln?��G>�W�>���.?��?���?�7�?x�?�b�?P6��4���h?�٣�N4@E��J�l��3ֿ�4���w?l��?�$�>	�ο��?�O4�v���˝�������>'����Ͼ:]���?��@�$���=��D���q��?���?�7�?V�7��?�k���|?����K���q~�g��q
¾'�?*�?(����t�a�?	�ο�ޗ>�G*�D��Ƅ]@B�M?:	?��?u��?_3_��߰�&)�JJ��z� �(em�\,|@m�+@���2=E=�ͫ�:]�2�"��ƀ>< ���G*��S�)N<Q���~�<�>�ӯ�L� �i:ο�3N��!����F s����?\�?�ZI�7ٛ?Ճ?^R@��?K��1��?�*�?UX�>E������+?��X��ҋ?(){?�����B�=�T�K���$�!���ဲ��z>�|����Ͼ	���cZ���ƀ>��q���x�bN����?�?��㾛|��*˳�9Ӟ�P6�=����{>%�M��$�Sux?�@�`,?-����%@��-���1���_?���K��:o��)N<A0>q=<�>��W�:]�P6��s��K�?�O4��S�)N<!��e�辛|��ԽG?���>�{��W?����'}�?��=BcW@��p?�+��@;�У�:]��C��OQ?̢����x����Չ�?Ȱ?"R�2=E=��>�?�{�K��d/��٣��q~�]?�VJ>k��2=E=u�_��'��>2����>�#���٣�ۺ/�E�����z房�z�?(��c�c?�x?=��1��?��-?UX�>�}Y�aF��u?�쮾��?���?���h�<b���z� �v����q(�B�h�������x>p׋��?����5@�7_��٣��3
?g�>>��n��쮾��w�:]�FwоI}'��$����$�����ى��,����X���Ͼ:]�%Tl����l.=�$��2ӿ��?.��?|�N��쮾X�ӿY1�?^�N� Ý>�#��K���$���@�ii>ŜE=ma?¬�>��-��7@1�
���?K��h~¾lk��"�I�U@�?0ҿ�ك�	�����1�$��?�#����x����g�Q]�n��� ?��˿ʇ@3�?��Q��?K��:o�����=2��>�+?a�}����?���p��a:@< ����x�f��!���!�����-��?�'�MF��%Tl����l.=�$��2ӿ��?.��?|�N��쮾X�ӿY1�?%Tl�$��?�#���$��	�Q����=���(�wm�?hݏ�ʇ@�]�?r�>�X���>6ң=Q>��;=cq�?-��w뿽L� �a�?��?�l.=��?��Q>y0F?���?���m�G���-��G?�3N����>z� ��S�˝��d|��x>Z���ԽG?��>^�N��J�=�>��x�;j<�q(�"G�?���>'0о\�Q@l�p=���?3
۾K�?��x��۠��}Y��y6�n�?��*��_>��>a�?�0����?�$��v���r��Q]��,���@;��=@���?���a�?����ǂ���聿)N<�r�O�$��@��Ͼl��?�$�>��>1��?w4˾*�@�}�?��?��7����ľ@:]��@�񱿜��?�G*�J��D���!����?�w��"Ԍ@�?�>俩3N���q��G*���m=�.?
�U?�$C��=%@u�_�E9�?�l3@��懡?��x�,H��D���ဲ��@�Ȱ�(��	����K@�-ֿN�&?�O4�}����>�X?]�P�7ٛ?���?�K@����?< ��w4˾�S�!����ii>������ ?���?���`yD��4E�z� ��sX>���>�f��EӾa�}��ك���t�2�"��-?����K���sX>lk��B�h��.�;���>��D����a�?1�
���?��x�"�>��@���wd|?R�/���w�uyE���R=/��>�7_���x��R��M7�z��?��7��4����<�?����� Ý>�M���$��:o��lk��A:�=lJ��`�?���=l��?����z;�������[��*�lk��y+��,��@��>У�:]�ʏ>K��W�?XD_>X�½Q>z��?O%F?�|���ӯ�:]�cZ��߹�@����z� ��S�˝�5�����7��˴?�ك���>�$�>�q~�N�&?h<U?
��>D���	�.�-`)�z[	�7ʏ>���?��d>6]�?�?z� ����@D���ɓ?�	�������M@D���P6��C�?�F���G*�:o��!�����?��=�s�V�7�:]��G?1�
�#[?K���R��q(��>>�9>��X�?�'�:]���>����G>��>f��0iG?y{@�Da?���hݏ���t�2�"�C> ����>w4˾�S�!���
9��z>�7�?w뿽D���)P=?z;����>XD_>;� ?Sux?a>�>�{�>��\���>��>>��-?����8,���$�lk�����⛿�XB@hݏ�l��?�C��S4?����ǂ������lk��aF��	���� ?�ԓ��ξ�]�?K���@	nF@v�����@�Ce?�Q�?�%m��͛?���?�G?��o���?N�G��$�@a>�>������?¬�>��>B�����|��?�٣���=r�5���3L���쮾u�_�
��?�]�?�񱿜��?��>���>E���U�?'0о��:]���d>Զ����?)4?}�����U�՜���|��¬�>���?k��/��>��q��٣�X�½���=��?`��E�(���?�ad@`yD�ۓ�?�W�>��=D���Y �-�?��\������>^�N��W?�ޗ>�M@��N@��@�@gd?-��(��:]�V��?=��ۓ�?k��>;� ?D���!��\�>�s��?Y1�?���/��>�!������+>A�_?���?��k���x>V�7�9Ӟ��?`yD���?��w�+?r��d|�O%F?�|����?�?���?�h�<|��?%�M�}�r�A�&����?'����Ͼ�'��k��I}'��$���ZW�Q> <~<5u����>u�_��4	������%?�����$���h�Q>!��l��2=E=�ӯ��ξ)*>��?��@���0�@���>�Ǔ>NAc��O��E@�'�����?�o��#[?h<U?,H��r��U�wd|?'��Ճ?�'��¦]�˹�?����ǂ����{@)N<!�����2=E=��g��?�3�?U�5�1��?w4˾q��>���>A:�=��?�Ȱ����4	���?�0��Q��?��[��.b�˝�ŀ�����a�}���g����?�{����?̢���k�@�BB?�Ο@G�@cq�?��?m�G�MF�����?��o�5@�٣���m=r�aF�� �?�%m�u�_�D���V��?	�ο��@?���h�r��>>�&��>��*�Q�@���>�x?����$���>'I�>!������ �?��X�(���ξ�$�>l�R�N�&?��x���ؾD�����r�~�Z�����?�r�?¦]���>b����_+@bN�g�Q]��,�� "?��Ͼ���>��1����M��K��h����?�ii>5u����>w뿽�4	�¦]������#���٣�v����q(�A�&�B���3��=(��:]�B����o��]��>��[��q~���@��y6��M�>�쮾7ʏ>uyE����G��>��q�K��۠�0iG?}>�?�ߧ��|��m�G��?�l��>/��>����K�򾟄$�E������]V?��\����4	�@��?K����@?w4˾,H���}Y�*�M�"R�c����W?�?k��/��>��q�0Sk@h~¾��?��@O%F?���>�ك���t��$�>1�
��h?K��T'F�r�J�l��y����x>7ʏ>l��?�C�`yD���G>�$��v����q(��r��,�� "?V�7���t���1�`yD��4E��٣������M7�	�.�
��|��m�G��)@���J�=��q�K��@=�.?�5C@q�M��쮾��Ͼ��-�%Tl���?̢��K�������?1䃾z房�|��(���"�>�@`yD�5@XD_>��+>���=ဲ�cq�?�Ȱ�hݏ����>Fwо���?̢��%�M�,H������d|�%K8�*�?f��>Y1�?�K@� Ý>�l.=���?�R�D���!�.?��]�2=E=7ʏ>O?k�����?QԾ�@�.>���>e�?�$T>a�}���W�aXM���-ֿ]��>��	�g��.?2��>��C> "?f��>�j�?���?3
۾�@��?ZW�r��>>�wd|?�%m���<�R4?TXi?�q~��3?�G*�X�½g���w�>O%F?��*���w�:]�l��>��]��>�O4�!�r������N%�z[	����?TXi?z;�1��?K�
Ao�$@��D@%��?��@@a�}���?�/��l��>��B?d/���?�sX>�q(��d�>���?��X�V�7�uyE�B�����G>�F���$��J������y�?�S?���>u�_�:]�l��>벪��ޗ>���}�!����f�!V��쮾0Г?Y1�?���?�4��W�?��>�.>)N< <~<wd|?'�� �@�>TXi?�0��W�?\�?J���q(��e�gÀ�G�K���W�l��?)P=?��G>��G>XD_>�@=lk���r��u?�w�����?�ξ(.���4���9l���[�ZW�lk����#���Ӥ2?7ʏ>�ξ2�"��w��JJ��%�M�:o��@Y'?e��>V�7�9Ӟ���>���?�$�z� �	�Q����aF�Ds�=E����=�ξ������B?�K@��x�j�n>A�_?A@ @�z>}j��m�G�	�������˹�?�#����=v�����D@E�?�v��<�>u�_�    �?��o�+�?�G*���r�*�M���6<c������>    �K@�1�
�̢��K���$�lk����_�v���x>�w?    a�?Զ����?XD_>�,;��q(�Y �&��>'��V�7�    V��?��>W�?�٣�h~¾�M7� <~<��?�|��(��    B����v�?�#����x�h~¾E���d|���2=E=��    �$�>#=�>�3?�k�@���F�?���?���?Z�����    2�"��w���$�w4˾,H���F�?aC+>X��:�S?��    ���l�R�+�?�G*�"�>E��5���Pܺ�@��>?�'�    �,@���� �?��=��=lk�� <~<-�?
]ڿhݏ�    2�"�G��>Y;y���?J��Sux?N��?�n2�'0о��g�    ���0����@��A?Z��@E�������z>'0о��)@    FwоU�5�&D�=K��	�Q���?ɓ?��C>�d?��Ͼ    �C��7�?�!���٣�(��=����U������� ?�ԓ�    2�"��3N��T�>��;!���q
¾��E�p׋�    �]�?K����@?N�G�T��?�R�?��;=tkV?<�>Gˣ?    *
dtype0
}
train_datasetTensorDataset%train_dataset/TensorToOutput/Constant*
output_shapes
:	�	*
Toutput_types
2
�(
$train_target/TensorToOutput/ConstantConst*�(
value�(B�(�	"�'  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  @@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  @@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  @@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  @@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  @@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  @@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  @@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  @@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  @@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@                                                                *
dtype0
w
train_targetTensorDataset$train_target/TensorToOutput/Constant*
output_shapes	
:�	*
Toutput_types
2
�
train_dataset/Zip
ZipDatasettrain_datasettrain_target*%
output_shapes
:	�	:�	*
N*
output_types
2
�
(Input_Input/Zip/Initializer/MakeIteratorMakeIteratortrain_dataset/ZipEstimator/Train/Model/Iterator*1
_class'
%#loc:@Estimator/Train/Model/Iterator "��[�     	^��	Vu��A�AJΠ

*
1.12.0-rc0ٟ

s
global_epochVarHandleOp*
shape: *
shared_nameglobal_epoch*
_class
 *
dtype0	*
	container 
�
Hglobal_epoch/Initializer/global_epoch/Initializer/ZerosInitializer/ShapeConst*
valueB *
_class
loc:@global_epoch*
dtype0
�
Qglobal_epoch/Initializer/global_epoch/Initializer/ZerosInitializer/Zeros/ConstantConst*
valueB	 "        *
_class
loc:@global_epoch*
dtype0	
�
Mglobal_epoch/Initializer/global_epoch/Initializer/ZerosInitializer/Zeros/FillFillHglobal_epoch/Initializer/global_epoch/Initializer/ZerosInitializer/ShapeQglobal_epoch/Initializer/global_epoch/Initializer/ZerosInitializer/Zeros/Constant*
T0	*

index_type0*
_class
loc:@global_epoch
�
!global_epoch/InitializationAssignAssignVariableOpglobal_epochMglobal_epoch/Initializer/global_epoch/Initializer/ZerosInitializer/Zeros/Fill*
dtype0	
l
global_epoch/Read/ReadVariableReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
>
global_epoch/IsInitializedVarIsInitializedOpglobal_epoch
t
&global_epoch/global_epoch/ReadVariableReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
q
global_stepVarHandleOp*
shared_nameglobal_step*
_class
 *
dtype0	*
	container *
shape: 
�
Fglobal_step/Initializer/global_step/Initializer/ZerosInitializer/ShapeConst*
valueB *
_class
loc:@global_step*
dtype0
�
Oglobal_step/Initializer/global_step/Initializer/ZerosInitializer/Zeros/ConstantConst*
valueB	 "        *
_class
loc:@global_step*
dtype0	
�
Kglobal_step/Initializer/global_step/Initializer/ZerosInitializer/Zeros/FillFillFglobal_step/Initializer/global_step/Initializer/ZerosInitializer/ShapeOglobal_step/Initializer/global_step/Initializer/ZerosInitializer/Zeros/Constant*
T0	*

index_type0*
_class
loc:@global_step
�
 global_step/InitializationAssignAssignVariableOpglobal_stepKglobal_step/Initializer/global_step/Initializer/ZerosInitializer/Zeros/Fill*
dtype0	
i
global_step/Read/ReadVariableReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
<
global_step/IsInitializedVarIsInitializedOpglobal_step
p
$global_step/global_step/ReadVariableReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
�
Estimator/Train/Model/IteratorIterator*
_class
 *
	container *
output_types
2*
shared_name *5
output_shapes$
":���������:���������
�
*Estimator/Train/Model/Input_Input/Zip/NextIteratorGetNextEstimator/Train/Model/Iterator*5
output_shapes$
":���������:���������*
output_types
2
P
Estimator/Train/Model/ShapeConst*
valueB"����   *
dtype0
�
#Estimator/Train/Model/Input/FlattenReshape*Estimator/Train/Model/Input_Input/Zip/NextEstimator/Train/Model/Shape*
T0*
Tshape0
�
\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/WeightsVarHandleOp*
_class
 *
dtype0*
	container *
shape
:@*m
shared_name^\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/ShapeConst*
valueB"   @   *o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/ConstantConst*
valueB
 "    *o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Constant_1Const*
valueB
 "  �?*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/RandomNormalInitializerRandomStandardNormal�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Shape*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0*
seed2 *

seed *
T0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/MultiplyMul�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/RandomNormalInitializer�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Constant_1*
T0*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/AddAdd�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Multiply�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Constant*
T0*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights
�
qInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/InitializationAssignAssignVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Add*
dtype0
�
nInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Read/ReadVariableReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
jInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/IsInitializedVarIsInitializedOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/ReadVariableReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
"Estimator/Train/Model/ReadVariableReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
YInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/BiasVarHandleOp*
	container *
shape:@*j
shared_name[YInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
_class
 *
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/ShapeConst*
valueB"@   *l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/ConstantConst*
valueB
 "    *l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Constant_1Const*
valueB
 "  �?*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/RandomNormalInitializerRandomStandardNormal�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Shape*
seed2 *

seed *
T0*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/MultiplyMul�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/RandomNormalInitializer�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Constant_1*
T0*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/AddAdd�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Multiply�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Constant*
T0*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias
�
nInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/InitializationAssignAssignVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Initializer/Add*
dtype0
�
kInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Read/ReadVariableReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
gInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/IsInitializedVarIsInitializedOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias
�
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/ReadVariableReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
$Estimator/Train/Model/ReadVariable_1ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
#Estimator/Train/Model/Linear/MatMulMatMul#Estimator/Train/Model/Input/Flatten"Estimator/Train/Model/ReadVariable*
T0*
transpose_a( *
transpose_b( 
�
$Estimator/Train/Model/Linear/AddBiasBiasAdd#Estimator/Train/Model/Linear/MatMul$Estimator/Train/Model/ReadVariable_1*
T0*
data_formatNHWC
c
,Estimator/Train/Model/ReLU/ReLU/PositivePartRelu$Estimator/Train/Model/Linear/AddBias*
T0
W
!Estimator/Train/Model/ReLU/NegateNeg$Estimator/Train/Model/Linear/AddBias*
T0
`
,Estimator/Train/Model/ReLU/ReLU/NegativePartRelu!Estimator/Train/Model/ReLU/Negate*
T0
P
#Estimator/Train/Model/ReLU/ConstantConst*
valueB
 "���=*
dtype0
�
#Estimator/Train/Model/ReLU/MultiplyMul#Estimator/Train/Model/ReLU/Constant,Estimator/Train/Model/ReLU/ReLU/NegativePart*
T0
�
#Estimator/Train/Model/ReLU/SubtractSub,Estimator/Train/Model/ReLU/ReLU/PositivePart#Estimator/Train/Model/ReLU/Multiply*
T0
�
@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/WeightsVarHandleOp*
	container *
shape
:@@*Q
shared_nameB@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
_class
 *
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/ShapeConst*
valueB"@   @   *S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/ConstantConst*
valueB
 "    *S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Constant_1Const*
valueB
 "  �?*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/RandomNormalInitializerRandomStandardNormal�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Shape*

seed *
T0*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0*
seed2 
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/MultiplyMul�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/RandomNormalInitializer�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Constant_1*
T0*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/AddAdd�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Multiply�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Constant*
T0*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights
�
UInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/InitializationAssignAssignVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Add*
dtype0
�
RInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Read/ReadVariableReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
NInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/IsInitializedVarIsInitializedOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/ReadVariableReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
$Estimator/Train/Model/ReadVariable_2ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/BiasVarHandleOp*
	container *
shape:@*N
shared_name?=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
_class
 *
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/ShapeConst*
valueB"@   *P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/ConstantConst*
valueB
 "    *P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Constant_1Const*
valueB
 "  �?*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/RandomNormalInitializerRandomStandardNormal�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Shape*
T0*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0*
seed2 *

seed 
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/MultiplyMul�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/RandomNormalInitializer�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Constant_1*
T0*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/AddAdd�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Multiply�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Constant*
T0*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias
�
RInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/InitializationAssignAssignVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Add*
dtype0
�
OInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Read/ReadVariableReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
KInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/IsInitializedVarIsInitializedOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/ReadVariableReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
$Estimator/Train/Model/ReadVariable_3ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
%Estimator/Train/Model/Linear_1/MatMulMatMul#Estimator/Train/Model/ReLU/Subtract$Estimator/Train/Model/ReadVariable_2*
T0*
transpose_a( *
transpose_b( 
�
&Estimator/Train/Model/Linear_1/AddBiasBiasAdd%Estimator/Train/Model/Linear_1/MatMul$Estimator/Train/Model/ReadVariable_3*
T0*
data_formatNHWC
g
.Estimator/Train/Model/ReLU_1/ReLU/PositivePartRelu&Estimator/Train/Model/Linear_1/AddBias*
T0
[
#Estimator/Train/Model/ReLU_1/NegateNeg&Estimator/Train/Model/Linear_1/AddBias*
T0
d
.Estimator/Train/Model/ReLU_1/ReLU/NegativePartRelu#Estimator/Train/Model/ReLU_1/Negate*
T0
R
%Estimator/Train/Model/ReLU_1/ConstantConst*
valueB
 "���=*
dtype0
�
%Estimator/Train/Model/ReLU_1/MultiplyMul%Estimator/Train/Model/ReLU_1/Constant.Estimator/Train/Model/ReLU_1/ReLU/NegativePart*
T0
�
%Estimator/Train/Model/ReLU_1/SubtractSub.Estimator/Train/Model/ReLU_1/ReLU/PositivePart%Estimator/Train/Model/ReLU_1/Multiply*
T0
�
(Input/Flatten/OutputLayer/Linear/WeightsVarHandleOp*9
shared_name*(Input/Flatten/OutputLayer/Linear/Weights*
_class
 *
dtype0*
	container *
shape
:@
�
oInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/ShapeConst*
valueB"@      *;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
rInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/ConstantConst*
valueB
 "    *;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
tInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/Constant_1Const*
valueB
 "  �?*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
�Input/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/RandomNormalInitializerRandomStandardNormaloInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/Shape*
T0*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0*
seed2 *

seed 
�
rInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/MultiplyMul�Input/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/RandomNormalInitializertInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/Constant_1*
T0*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights
�
mInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/AddAddrInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/MultiplyrInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/Constant*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
T0
�
=Input/Flatten/OutputLayer/Linear/Weights/InitializationAssignAssignVariableOp(Input/Flatten/OutputLayer/Linear/WeightsmInput/Flatten/OutputLayer/Linear/Weights/Initializer/Input/Flatten/OutputLayer/Linear/Weights/Initializer/Add*
dtype0
�
:Input/Flatten/OutputLayer/Linear/Weights/Read/ReadVariableReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
v
6Input/Flatten/OutputLayer/Linear/Weights/IsInitializedVarIsInitializedOp(Input/Flatten/OutputLayer/Linear/Weights
�
^Input/Flatten/OutputLayer/Linear/Weights/Input/Flatten/OutputLayer/Linear/Weights/ReadVariableReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
$Estimator/Train/Model/ReadVariable_4ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
%Input/Flatten/OutputLayer/Linear/BiasVarHandleOp*
	container *
shape:*6
shared_name'%Input/Flatten/OutputLayer/Linear/Bias*
_class
 *
dtype0
�
iInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/ShapeConst*
valueB"   *8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
lInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/ConstantConst*
valueB
 "    *8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
nInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/Constant_1Const*
valueB
 "  �?*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
{Input/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/RandomNormalInitializerRandomStandardNormaliInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/Shape*
seed2 *

seed *
T0*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
lInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/MultiplyMul{Input/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/RandomNormalInitializernInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/Constant_1*
T0*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias
�
gInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/AddAddlInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/MultiplylInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/Constant*
T0*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias
�
:Input/Flatten/OutputLayer/Linear/Bias/InitializationAssignAssignVariableOp%Input/Flatten/OutputLayer/Linear/BiasgInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/Add*
dtype0
�
7Input/Flatten/OutputLayer/Linear/Bias/Read/ReadVariableReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
p
3Input/Flatten/OutputLayer/Linear/Bias/IsInitializedVarIsInitializedOp%Input/Flatten/OutputLayer/Linear/Bias
�
XInput/Flatten/OutputLayer/Linear/Bias/Input/Flatten/OutputLayer/Linear/Bias/ReadVariableReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
$Estimator/Train/Model/ReadVariable_5ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
%Estimator/Train/Model/Linear_2/MatMulMatMul%Estimator/Train/Model/ReLU_1/Subtract$Estimator/Train/Model/ReadVariable_4*
T0*
transpose_a( *
transpose_b( 
�
&Estimator/Train/Model/Linear_2/AddBiasBiasAdd%Estimator/Train/Model/Linear_2/MatMul$Estimator/Train/Model/ReadVariable_5*
T0*
data_formatNHWC
�
Estimator/Train/Model/SubtractSub&Estimator/Train/Model/Linear_2/AddBias,Estimator/Train/Model/Input_Input/Zip/Next:1*
T0
P
Estimator/Train/Model/Loss/L2L2LossEstimator/Train/Model/Subtract*
T0
Z
1Estimator/Train/Model/Gradients/Gradients_0/ShapeConst*
valueB *
dtype0
f
9Estimator/Train/Model/Gradients/Gradients_0/Ones/ConstantConst*
valueB
 "  �?*
dtype0
�
5Estimator/Train/Model/Gradients/Gradients_0/Ones/FillFill1Estimator/Train/Model/Gradients/Gradients_0/Shape9Estimator/Train/Model/Gradients/Gradients_0/Ones/Constant*

index_type0*
T0
�
NEstimator/Train/Model/Gradients/Estimator/Train/Model/Loss/L2Gradient/MultiplyMulEstimator/Train/Model/Subtract5Estimator/Train/Model/Gradients/Gradients_0/Ones/Fill*
T0
�
LEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/ShapeShape&Estimator/Train/Model/Linear_2/AddBias*
T0*
out_type0
�
NEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Shape_1Shape,Estimator/Train/Model/Input_Input/Zip/Next:1*
out_type0*
T0
�
aEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/BroadcastGradientArgumentsBroadcastGradientArgsLEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/ShapeNEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Shape_1*
T0
�
JEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/SumSumNEstimator/Train/Model/Gradients/Estimator/Train/Model/Loss/L2Gradient/MultiplyaEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/BroadcastGradientArguments*

Tidx0*
	keep_dims( *
T0
�
NEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/ReshapeReshapeJEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/SumLEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Shape*
T0*
Tshape0*
_class
 
�
LEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Sum_1SumNEstimator/Train/Model/Gradients/Estimator/Train/Model/Loss/L2Gradient/MultiplycEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/BroadcastGradientArguments:1*

Tidx0*
	keep_dims( *
T0
�
MEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/NegateNegLEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Sum_1*
T0
�
PEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Reshape_1ReshapeMEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/NegateNEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Shape_1*
T0*
_class
 *
Tshape0
�
REstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/GroupNoOpO^Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/ReshapeQ^Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Reshape_1
�
vEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityNEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/ReshapeS^Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/Group*
T0*a
_classW
USloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Reshape
�
xEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityPEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Reshape_1S^Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/Group*
T0*c
_classY
WUloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Reshape_1
�
^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/BiasAddGradientBiasAddGradvEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/WithControlDependencies/Identity/Identity*
T0*
_class
 *
data_formatNHWC
�
ZEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/GroupNoOp_^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/BiasAddGradientw^Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/WithControlDependencies/Identity/Identity
�
~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityvEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/WithControlDependencies/Identity/Identity[^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/Group*
T0*�
_class�
�Sloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Reshape{loc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Tuple/WithControlDependencies/Identity/Identity
�
�Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentity^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/BiasAddGradient[^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/Group*
T0*q
_classg
ecloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/BiasAddGradient
�
TEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMulMatMul~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity$Estimator/Train/Model/ReadVariable_4*
_class
 *
transpose_a( *
transpose_b(*
T0
�
VEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul_1MatMul%Estimator/Train/Model/ReLU_1/Subtract~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity*
T0*
_class
 *
transpose_a(*
transpose_b( 
�
YEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/GroupNoOpU^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMulW^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul_1
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityTEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMulZ^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/Group*
T0*g
_class]
[Yloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul
�
Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityVEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul_1Z^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/Group*
T0*i
_class_
][loc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul_1
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/ShapeShape.Estimator/Train/Model/ReLU_1/ReLU/PositivePart*
T0*
out_type0
�
UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Shape_1Shape%Estimator/Train/Model/ReLU_1/Multiply*
T0*
out_type0
�
hEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/BroadcastGradientArgumentsBroadcastGradientArgsSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/ShapeUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Shape_1*
T0
�
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/SumSum}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityhEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/BroadcastGradientArguments*

Tidx0*
	keep_dims( *
T0
�
UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/ReshapeReshapeQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/SumSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Shape*
T0*
_class
 *
Tshape0
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Sum_1Sum}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityjEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/BroadcastGradientArguments:1*

Tidx0*
	keep_dims( *
T0
�
TEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/NegateNegSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Sum_1*
T0
�
WEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Reshape_1ReshapeTEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/NegateUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Shape_1*
T0*
_class
 *
Tshape0
�
YEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/GroupNoOpV^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/ReshapeX^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Reshape_1
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/ReshapeZ^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/Group*h
_class^
\Zloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Reshape*
T0
�
Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityWEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Reshape_1Z^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/Group*
T0*j
_class`
^\loc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Reshape_1
�
cEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/ReLU/PositivePartGradient/ReLUGradientReluGrad}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/WithControlDependencies/Identity/Identity.Estimator/Train/Model/ReLU_1/ReLU/PositivePart*
T0*
_class
 
|
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/ShapeConst*
valueB *
dtype0
�
UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Shape_1Shape.Estimator/Train/Model/ReLU_1/ReLU/NegativePart*
out_type0*
T0
�
hEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/BroadcastGradientArgumentsBroadcastGradientArgsSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/ShapeUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Shape_1*
T0
�
VEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/MultiplyMulEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/WithControlDependencies_1/Identity/Identity.Estimator/Train/Model/ReLU_1/ReLU/NegativePart*
T0
�
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/SumSumVEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/MultiplyhEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/BroadcastGradientArguments*

Tidx0*
	keep_dims( *
T0
�
UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/ReshapeReshapeQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/SumSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Shape*
T0*
_class
 *
Tshape0
�
XEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Multiply_1Mul%Estimator/Train/Model/ReLU_1/ConstantEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/SubtractGradient/Tuple/WithControlDependencies_1/Identity/Identity*
T0
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Sum_1SumXEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Multiply_1jEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/BroadcastGradientArguments:1*

Tidx0*
	keep_dims( *
T0
�
WEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Reshape_1ReshapeSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Sum_1UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Shape_1*
T0*
_class
 *
Tshape0
�
YEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Tuple/GroupNoOpV^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/ReshapeX^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Reshape_1
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/ReshapeZ^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Tuple/Group*
T0*h
_class^
\Zloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Reshape
�
Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityWEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Reshape_1Z^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Tuple/Group*
T0*j
_class`
^\loc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Reshape_1
�
cEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/ReLU/NegativePartGradient/ReLUGradientReluGradEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/Tuple/WithControlDependencies_1/Identity/Identity.Estimator/Train/Model/ReLU_1/ReLU/NegativePart*
T0
�
REstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/NegateGradient/NegateNegcEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/ReLU/NegativePartGradient/ReLUGradient*
T0
�
$Estimator/Train/Model/Gradients/AddNAddNcEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/ReLU/PositivePartGradient/ReLUGradientREstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/NegateGradient/Negate*v
_classl
jhloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/ReLU/PositivePartGradient/ReLUGradient*
N*
T0
t
+Estimator/Train/Model/Gradients/AddN_1/AddNIdentity$Estimator/Train/Model/Gradients/AddN*
T0*
_class
 
�
^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/BiasAddGradientBiasAddGrad+Estimator/Train/Model/Gradients/AddN_1/AddN*
T0*
_class
 *
data_formatNHWC
�
ZEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/GroupNoOp,^Estimator/Train/Model/Gradients/AddN_1/AddN_^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/BiasAddGradient
�
~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/WithControlDependencies/Identity/IdentityIdentity+Estimator/Train/Model/Gradients/AddN_1/AddN[^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/Group*
T0*>
_class4
20loc:@Estimator/Train/Model/Gradients/AddN_1/AddN
�
�Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentity^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/BiasAddGradient[^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/Group*
T0*q
_classg
ecloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/BiasAddGradient
�
TEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMulMatMul~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity$Estimator/Train/Model/ReadVariable_2*
transpose_a( *
transpose_b(*
T0*
_class
 
�
VEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMul_1MatMul#Estimator/Train/Model/ReLU/Subtract~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity*
T0*
_class
 *
transpose_a(*
transpose_b( 
�
YEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/GroupNoOpU^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMulW^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMul_1
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityTEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMulZ^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/Group*g
_class]
[Yloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMul*
T0
�
Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityVEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMul_1Z^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/Group*
T0*i
_class_
][loc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMul_1
�
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/ShapeShape,Estimator/Train/Model/ReLU/ReLU/PositivePart*
T0*
out_type0
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Shape_1Shape#Estimator/Train/Model/ReLU/Multiply*
T0*
out_type0
�
fEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/BroadcastGradientArgumentsBroadcastGradientArgsQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/ShapeSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Shape_1*
T0
�
OEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/SumSum}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityfEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/BroadcastGradientArguments*

Tidx0*
	keep_dims( *
T0
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/ReshapeReshapeOEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/SumQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Shape*
T0*
_class
 *
Tshape0
�
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Sum_1Sum}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityhEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/BroadcastGradientArguments:1*

Tidx0*
	keep_dims( *
T0
�
REstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/NegateNegQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Sum_1*
T0
�
UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Reshape_1ReshapeREstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/NegateSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Shape_1*
_class
 *
Tshape0*
T0
�
WEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/GroupNoOpT^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/ReshapeV^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Reshape_1
�
{Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/WithControlDependencies/Identity/IdentityIdentitySEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/ReshapeX^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/Group*
T0*f
_class\
ZXloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Reshape
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Reshape_1X^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/Group*h
_class^
\Zloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Reshape_1*
T0
�
aEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/ReLU/PositivePartGradient/ReLUGradientReluGrad{Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/WithControlDependencies/Identity/Identity,Estimator/Train/Model/ReLU/ReLU/PositivePart*
T0*
_class
 
z
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/ShapeConst*
valueB *
dtype0
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Shape_1Shape,Estimator/Train/Model/ReLU/ReLU/NegativePart*
T0*
out_type0
�
fEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/BroadcastGradientArgumentsBroadcastGradientArgsQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/ShapeSEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Shape_1*
T0
�
TEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/MultiplyMul}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/WithControlDependencies_1/Identity/Identity,Estimator/Train/Model/ReLU/ReLU/NegativePart*
T0
�
OEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/SumSumTEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/MultiplyfEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/BroadcastGradientArguments*
T0*

Tidx0*
	keep_dims( 
�
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/ReshapeReshapeOEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/SumQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Shape*
T0*
_class
 *
Tshape0
�
VEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Multiply_1Mul#Estimator/Train/Model/ReLU/Constant}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/WithControlDependencies_1/Identity/Identity*
T0
�
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Sum_1SumVEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Multiply_1hEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/BroadcastGradientArguments:1*

Tidx0*
	keep_dims( *
T0
�
UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Reshape_1ReshapeQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Sum_1SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Shape_1*
T0*
_class
 *
Tshape0
�
WEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Tuple/GroupNoOpT^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/ReshapeV^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Reshape_1
�
{Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Tuple/WithControlDependencies/Identity/IdentityIdentitySEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/ReshapeX^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Tuple/Group*
T0*f
_class\
ZXloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Reshape
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityUEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Reshape_1X^Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Tuple/Group*
T0*h
_class^
\Zloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Reshape_1
�
aEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/ReLU/NegativePartGradient/ReLUGradientReluGrad}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Tuple/WithControlDependencies_1/Identity/Identity,Estimator/Train/Model/ReLU/ReLU/NegativePart*
T0
�
PEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/NegateGradient/NegateNegaEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/ReLU/NegativePartGradient/ReLUGradient*
T0
�
&Estimator/Train/Model/Gradients/AddN_2AddNaEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/ReLU/PositivePartGradient/ReLUGradientPEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/NegateGradient/Negate*
T0*t
_classj
hfloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/ReLU/PositivePartGradient/ReLUGradient*
N
v
+Estimator/Train/Model/Gradients/AddN_3/AddNIdentity&Estimator/Train/Model/Gradients/AddN_2*
T0*
_class
 
�
\Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/BiasAddGradientBiasAddGrad+Estimator/Train/Model/Gradients/AddN_3/AddN*
T0*
_class
 *
data_formatNHWC
�
XEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/GroupNoOp,^Estimator/Train/Model/Gradients/AddN_3/AddN]^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/BiasAddGradient
�
|Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/WithControlDependencies/Identity/IdentityIdentity+Estimator/Train/Model/Gradients/AddN_3/AddNY^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/Group*
T0*>
_class4
20loc:@Estimator/Train/Model/Gradients/AddN_3/AddN
�
~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentity\Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/BiasAddGradientY^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/Group*
T0*o
_classe
caloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/BiasAddGradient
�
REstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMulMatMul|Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity"Estimator/Train/Model/ReadVariable*
T0*
_class
 *
transpose_a( *
transpose_b(
�
TEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMul_1MatMul#Estimator/Train/Model/Input/Flatten|Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity*
transpose_a(*
transpose_b( *
T0*
_class
 
�
WEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/Tuple/GroupNoOpS^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMulU^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMul_1
�
{Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityREstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMulX^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/Tuple/Group*
T0*e
_class[
YWloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMul
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityTEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMul_1X^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/Tuple/Group*
T0*g
_class]
[Yloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMul_1
_
2Estimator/Train/Model/GradientDescent/LearningRateConst*
valueB
 "
�#<*
dtype0
�
�Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/GradientDescent/ApplyDenseResourceApplyGradientDescent@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights2Estimator/Train/Model/GradientDescent/LearningRateEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/Tuple/WithControlDependencies_1/Identity/Identity*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
use_locking( *
T0
�
mEstimator/Train/Model/GradientDescent/Update/Input/Flatten/OutputLayer/Linear/Bias/GradientDescent/ApplyDenseResourceApplyGradientDescent%Input/Flatten/OutputLayer/Linear/Bias2Estimator/Train/Model/GradientDescent/LearningRate�Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/Identity*
use_locking( *
T0*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias
�
�Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/GradientDescent/ApplyDenseResourceApplyGradientDescentYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias2Estimator/Train/Model/GradientDescent/LearningRate~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/Identity*
use_locking( *
T0*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias
�
�Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/GradientDescent/ApplyDenseResourceApplyGradientDescent\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights2Estimator/Train/Model/GradientDescent/LearningRate}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/Tuple/WithControlDependencies_1/Identity/Identity*
use_locking( *
T0*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights
�
�Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/GradientDescent/ApplyDenseResourceApplyGradientDescent=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias2Estimator/Train/Model/GradientDescent/LearningRate�Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/Identity*
use_locking( *
T0*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias
�
pEstimator/Train/Model/GradientDescent/Update/Input/Flatten/OutputLayer/Linear/Weights/GradientDescent/ApplyDenseResourceApplyGradientDescent(Input/Flatten/OutputLayer/Linear/Weights2Estimator/Train/Model/GradientDescent/LearningRateEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/WithControlDependencies_1/Identity/Identity*
use_locking( *
T0*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights
�
,Estimator/Train/Model/GradientDescent/FinishNoOp�^Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/GradientDescent/ApplyDense�^Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/GradientDescent/ApplyDense�^Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/GradientDescent/ApplyDense�^Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/GradientDescent/ApplyDensen^Estimator/Train/Model/GradientDescent/Update/Input/Flatten/OutputLayer/Linear/Bias/GradientDescent/ApplyDenseq^Estimator/Train/Model/GradientDescent/Update/Input/Flatten/OutputLayer/Linear/Weights/GradientDescent/ApplyDense
�
+Estimator/Train/Model/GradientDescent/ShapeConst-^Estimator/Train/Model/GradientDescent/Finish*
valueB *
_class
loc:@global_step*
dtype0
�
3Estimator/Train/Model/GradientDescent/Ones/ConstantConst-^Estimator/Train/Model/GradientDescent/Finish*
valueB	 "       *
_class
loc:@global_step*
dtype0	
�
/Estimator/Train/Model/GradientDescent/Ones/FillFill+Estimator/Train/Model/GradientDescent/Shape3Estimator/Train/Model/GradientDescent/Ones/Constant*
T0	*

index_type0*
_class
loc:@global_step
�
5Estimator/Train/Model/GradientDescent/GradientDescentAssignAddVariableOpglobal_step/Estimator/Train/Model/GradientDescent/Ones/Fill*
_class
loc:@global_step*
dtype0	
�
Estimator/Infer/Model/IteratorIterator*
	container *
output_types
2*
shared_name *&
output_shapes
:���������
�
 Estimator/Infer/Model/Input/NextIteratorGetNextEstimator/Infer/Model/Iterator*&
output_shapes
:���������*
output_types
2
P
Estimator/Infer/Model/ShapeConst*
valueB"����   *
dtype0
�
#Estimator/Infer/Model/Input/FlattenReshape Estimator/Infer/Model/Input/NextEstimator/Infer/Model/Shape*
Tshape0*
T0
�
"Estimator/Infer/Model/ReadVariableReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
$Estimator/Infer/Model/ReadVariable_1ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
#Estimator/Infer/Model/Linear/MatMulMatMul#Estimator/Infer/Model/Input/Flatten"Estimator/Infer/Model/ReadVariable*
T0*
transpose_a( *
transpose_b( 
�
$Estimator/Infer/Model/Linear/AddBiasBiasAdd#Estimator/Infer/Model/Linear/MatMul$Estimator/Infer/Model/ReadVariable_1*
data_formatNHWC*
T0
c
,Estimator/Infer/Model/ReLU/ReLU/PositivePartRelu$Estimator/Infer/Model/Linear/AddBias*
T0
W
!Estimator/Infer/Model/ReLU/NegateNeg$Estimator/Infer/Model/Linear/AddBias*
T0
`
,Estimator/Infer/Model/ReLU/ReLU/NegativePartRelu!Estimator/Infer/Model/ReLU/Negate*
T0
P
#Estimator/Infer/Model/ReLU/ConstantConst*
valueB
 "���=*
dtype0
�
#Estimator/Infer/Model/ReLU/MultiplyMul#Estimator/Infer/Model/ReLU/Constant,Estimator/Infer/Model/ReLU/ReLU/NegativePart*
T0
�
#Estimator/Infer/Model/ReLU/SubtractSub,Estimator/Infer/Model/ReLU/ReLU/PositivePart#Estimator/Infer/Model/ReLU/Multiply*
T0
�
$Estimator/Infer/Model/ReadVariable_2ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
$Estimator/Infer/Model/ReadVariable_3ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
%Estimator/Infer/Model/Linear_1/MatMulMatMul#Estimator/Infer/Model/ReLU/Subtract$Estimator/Infer/Model/ReadVariable_2*
T0*
transpose_a( *
transpose_b( 
�
&Estimator/Infer/Model/Linear_1/AddBiasBiasAdd%Estimator/Infer/Model/Linear_1/MatMul$Estimator/Infer/Model/ReadVariable_3*
T0*
data_formatNHWC
g
.Estimator/Infer/Model/ReLU_1/ReLU/PositivePartRelu&Estimator/Infer/Model/Linear_1/AddBias*
T0
[
#Estimator/Infer/Model/ReLU_1/NegateNeg&Estimator/Infer/Model/Linear_1/AddBias*
T0
d
.Estimator/Infer/Model/ReLU_1/ReLU/NegativePartRelu#Estimator/Infer/Model/ReLU_1/Negate*
T0
R
%Estimator/Infer/Model/ReLU_1/ConstantConst*
valueB
 "���=*
dtype0
�
%Estimator/Infer/Model/ReLU_1/MultiplyMul%Estimator/Infer/Model/ReLU_1/Constant.Estimator/Infer/Model/ReLU_1/ReLU/NegativePart*
T0
�
%Estimator/Infer/Model/ReLU_1/SubtractSub.Estimator/Infer/Model/ReLU_1/ReLU/PositivePart%Estimator/Infer/Model/ReLU_1/Multiply*
T0
�
$Estimator/Infer/Model/ReadVariable_4ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
$Estimator/Infer/Model/ReadVariable_5ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
%Estimator/Infer/Model/Linear_2/MatMulMatMul%Estimator/Infer/Model/ReLU_1/Subtract$Estimator/Infer/Model/ReadVariable_4*
T0*
transpose_a( *
transpose_b( 
�
&Estimator/Infer/Model/Linear_2/AddBiasBiasAdd%Estimator/Infer/Model/Linear_2/MatMul$Estimator/Infer/Model/ReadVariable_5*
T0*
data_formatNHWC
�
!Estimator/Evaluate/Model/IteratorIterator*
	container *
output_types
2*
shared_name *5
output_shapes$
":���������:���������
�
-Estimator/Evaluate/Model/Input_Input/Zip/NextIteratorGetNext!Estimator/Evaluate/Model/Iterator*
output_types
2*5
output_shapes$
":���������:���������
S
Estimator/Evaluate/Model/ShapeConst*
valueB"����   *
dtype0
�
&Estimator/Evaluate/Model/Input/FlattenReshape-Estimator/Evaluate/Model/Input_Input/Zip/NextEstimator/Evaluate/Model/Shape*
T0*
Tshape0
�
%Estimator/Evaluate/Model/ReadVariableReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
'Estimator/Evaluate/Model/ReadVariable_1ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
&Estimator/Evaluate/Model/Linear/MatMulMatMul&Estimator/Evaluate/Model/Input/Flatten%Estimator/Evaluate/Model/ReadVariable*
T0*
transpose_a( *
transpose_b( 
�
'Estimator/Evaluate/Model/Linear/AddBiasBiasAdd&Estimator/Evaluate/Model/Linear/MatMul'Estimator/Evaluate/Model/ReadVariable_1*
T0*
data_formatNHWC
i
/Estimator/Evaluate/Model/ReLU/ReLU/PositivePartRelu'Estimator/Evaluate/Model/Linear/AddBias*
T0
]
$Estimator/Evaluate/Model/ReLU/NegateNeg'Estimator/Evaluate/Model/Linear/AddBias*
T0
f
/Estimator/Evaluate/Model/ReLU/ReLU/NegativePartRelu$Estimator/Evaluate/Model/ReLU/Negate*
T0
S
&Estimator/Evaluate/Model/ReLU/ConstantConst*
valueB
 "���=*
dtype0
�
&Estimator/Evaluate/Model/ReLU/MultiplyMul&Estimator/Evaluate/Model/ReLU/Constant/Estimator/Evaluate/Model/ReLU/ReLU/NegativePart*
T0
�
&Estimator/Evaluate/Model/ReLU/SubtractSub/Estimator/Evaluate/Model/ReLU/ReLU/PositivePart&Estimator/Evaluate/Model/ReLU/Multiply*
T0
�
'Estimator/Evaluate/Model/ReadVariable_2ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
'Estimator/Evaluate/Model/ReadVariable_3ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
(Estimator/Evaluate/Model/Linear_1/MatMulMatMul&Estimator/Evaluate/Model/ReLU/Subtract'Estimator/Evaluate/Model/ReadVariable_2*
transpose_a( *
transpose_b( *
T0
�
)Estimator/Evaluate/Model/Linear_1/AddBiasBiasAdd(Estimator/Evaluate/Model/Linear_1/MatMul'Estimator/Evaluate/Model/ReadVariable_3*
data_formatNHWC*
T0
m
1Estimator/Evaluate/Model/ReLU_1/ReLU/PositivePartRelu)Estimator/Evaluate/Model/Linear_1/AddBias*
T0
a
&Estimator/Evaluate/Model/ReLU_1/NegateNeg)Estimator/Evaluate/Model/Linear_1/AddBias*
T0
j
1Estimator/Evaluate/Model/ReLU_1/ReLU/NegativePartRelu&Estimator/Evaluate/Model/ReLU_1/Negate*
T0
U
(Estimator/Evaluate/Model/ReLU_1/ConstantConst*
valueB
 "���=*
dtype0
�
(Estimator/Evaluate/Model/ReLU_1/MultiplyMul(Estimator/Evaluate/Model/ReLU_1/Constant1Estimator/Evaluate/Model/ReLU_1/ReLU/NegativePart*
T0
�
(Estimator/Evaluate/Model/ReLU_1/SubtractSub1Estimator/Evaluate/Model/ReLU_1/ReLU/PositivePart(Estimator/Evaluate/Model/ReLU_1/Multiply*
T0
�
'Estimator/Evaluate/Model/ReadVariable_4ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
'Estimator/Evaluate/Model/ReadVariable_5ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
(Estimator/Evaluate/Model/Linear_2/MatMulMatMul(Estimator/Evaluate/Model/ReLU_1/Subtract'Estimator/Evaluate/Model/ReadVariable_4*
transpose_a( *
transpose_b( *
T0
�
)Estimator/Evaluate/Model/Linear_2/AddBiasBiasAdd(Estimator/Evaluate/Model/Linear_2/MatMul'Estimator/Evaluate/Model/ReadVariable_5*
T0*
data_formatNHWC
m
	eval_stepVarHandleOp*
shape: *
shared_name	eval_step*
_class
 *
dtype0	*
	container 
�
Beval_step/Initializer/eval_step/Initializer/ZerosInitializer/ShapeConst*
valueB *
_class
loc:@eval_step*
dtype0
�
Keval_step/Initializer/eval_step/Initializer/ZerosInitializer/Zeros/ConstantConst*
valueB	 "        *
_class
loc:@eval_step*
dtype0	
�
Geval_step/Initializer/eval_step/Initializer/ZerosInitializer/Zeros/FillFillBeval_step/Initializer/eval_step/Initializer/ZerosInitializer/ShapeKeval_step/Initializer/eval_step/Initializer/ZerosInitializer/Zeros/Constant*
T0	*

index_type0*
_class
loc:@eval_step
�
eval_step/InitializationAssignAssignVariableOp	eval_stepGeval_step/Initializer/eval_step/Initializer/ZerosInitializer/Zeros/Fill*
dtype0	
c
eval_step/Read/ReadVariableReadVariableOp	eval_step*
_class
loc:@eval_step*
dtype0	
8
eval_step/IsInitializedVarIsInitializedOp	eval_step
h
 eval_step/eval_step/ReadVariableReadVariableOp	eval_step*
_class
loc:@eval_step*
dtype0	
L
Estimator/Evaluate/ConstantConst*
valueB	 "       *
dtype0	
h
Estimator/Evaluate/AssignAddAssignAddVariableOp	eval_stepEstimator/Evaluate/Constant*
dtype0	
g
Estimator/Evaluate/AssignAdd_1ReadVariableOp	eval_step^Estimator/Evaluate/AssignAdd*
dtype0	
A
Estimator/Evaluate/GroupNoOp^Estimator/Evaluate/AssignAdd_1
b
Estimator/ReadVariableReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
J
Saver/ConstantConst*
_class
 *
valueB Bmodel*
dtype0
�
Saver/ReadVariableReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/ReadVariable_1ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/ReadVariable_2ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/ReadVariable_3ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/ReadVariable_4ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
Saver/ReadVariable_5ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
Saver/ReadVariable_6ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/ReadVariable_7ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/ReadVariable_8ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
Saver/ReadVariable_9ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
a
Saver/ReadVariable_10ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
a
Saver/ReadVariable_11ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
�
Saver/ReadVariable_12ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
Saver/ReadVariable_13ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
c
Saver/ReadVariable_14ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
c
Saver/ReadVariable_15ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
h
Saver/Constant_1Const*@
value7B5 B/_temp_fa9f4401-8f8c-4b4d-9131-e82ab6e86d17/part*
dtype0
Z
Saver/StringJoin
StringJoinSaver/ConstantSaver/Constant_1*
	separator *
N
L
Saver/Constant_2Const"/device:CPU:0*
valueB
 "    *
dtype0
L
Saver/Constant_3Const"/device:CPU:0*
valueB
 "   *
dtype0
{
Saver/ShardedFilenameShardedFilenameSaver/StringJoinSaver/Constant_2Saver/Constant_3"/device:CPU:0*
_class
 
�
Saver/ReadVariable_16ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/ReadVariable_17ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
a
Saver/ReadVariable_18ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
�
Saver/ReadVariable_19ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
Saver/ReadVariable_20ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
Saver/ReadVariable_21ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/ReadVariable_22ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
c
Saver/ReadVariable_23ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
�
Saver/Constant_4Const"/device:CPU:0*�
value�B�Bglobal_stepB%Input/Flatten/OutputLayer/Linear/BiasB=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/BiasB@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/WeightsBglobal_epochB(Input/Flatten/OutputLayer/Linear/WeightsBYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/BiasB\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
Z
Saver/Constant_5Const"/device:CPU:0*#
valueBB B B B B B B B *
dtype0
�

Saver/SaveSaveV2Saver/ShardedFilenameSaver/Constant_4Saver/Constant_5Saver/ReadVariable_18Saver/ReadVariable_20Saver/ReadVariable_16Saver/ReadVariable_22Saver/ReadVariable_23Saver/ReadVariable_19Saver/ReadVariable_17Saver/ReadVariable_21"/device:CPU:0*
dtypes

2		
�
/Saver/WithControlDependencies/Identity/IdentityIdentitySaver/ShardedFilename^Saver/Save"/device:CPU:0*
T0*(
_class
loc:@Saver/ShardedFilename
}
Saver/ShapeConst0^Saver/WithControlDependencies/Identity/Identity"/device:CPU:0*
valueB"   *
dtype0
b
Saver/ReshapeReshapeSaver/ShardedFilenameSaver/Shape"/device:CPU:0*
T0*
Tshape0
s
Saver/MergeV2CheckpointsMergeV2CheckpointsSaver/ReshapeSaver/Constant"/device:CPU:0*
delete_old_dirs(
�
1Saver/WithControlDependencies_1/Identity/IdentityIdentitySaver/Constant^Saver/MergeV2Checkpoints0^Saver/WithControlDependencies/Identity/Identity*!
_class
loc:@Saver/Constant*
T0
�
Saver/ReadVariable_24ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/ReadVariable_25ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/ReadVariable_26ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/Constant_6Const"/device:CPU:0*q
valuehBfB\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
L
Saver/Constant_7Const"/device:CPU:0*
valueB
B *
dtype0
n
Saver/Restore	RestoreV2Saver/ConstantSaver/Constant_6Saver/Constant_7"/device:CPU:0*
dtypes
2
J
Saver/Identity/IdentityIdentitySaver/Restore"/device:CPU:0*
T0
�
Saver/AssignVariableAssignVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/WeightsSaver/Identity/Identity"/device:CPU:0*
dtype0
�
Saver/ReadVariable_27ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/ReadVariable_28ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/ReadVariable_29ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/Constant_8Const"/device:CPU:0*R
valueIBGB=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
L
Saver/Constant_9Const"/device:CPU:0*
valueB
B *
dtype0
p
Saver/Restore_1	RestoreV2Saver/ConstantSaver/Constant_8Saver/Constant_9"/device:CPU:0*
dtypes
2
N
Saver/Identity_1/IdentityIdentitySaver/Restore_1"/device:CPU:0*
T0
�
Saver/AssignVariable_1AssignVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/BiasSaver/Identity_1/Identity"/device:CPU:0*
dtype0
�
Saver/ReadVariable_30ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
Saver/ReadVariable_31ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
Saver/ReadVariable_32ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
r
Saver/Constant_10Const"/device:CPU:0*:
value1B/B%Input/Flatten/OutputLayer/Linear/Bias*
dtype0
M
Saver/Constant_11Const"/device:CPU:0*
valueB
B *
dtype0
r
Saver/Restore_2	RestoreV2Saver/ConstantSaver/Constant_10Saver/Constant_11"/device:CPU:0*
dtypes
2
N
Saver/Identity_2/IdentityIdentitySaver/Restore_2"/device:CPU:0*
T0
�
Saver/AssignVariable_2AssignVariableOp%Input/Flatten/OutputLayer/Linear/BiasSaver/Identity_2/Identity"/device:CPU:0*
dtype0
�
Saver/ReadVariable_33ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/ReadVariable_34ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/ReadVariable_35ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/Constant_12Const"/device:CPU:0*U
valueLBJB@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
M
Saver/Constant_13Const"/device:CPU:0*
valueB
B *
dtype0
r
Saver/Restore_3	RestoreV2Saver/ConstantSaver/Constant_12Saver/Constant_13"/device:CPU:0*
dtypes
2
N
Saver/Identity_3/IdentityIdentitySaver/Restore_3"/device:CPU:0*
T0
�
Saver/AssignVariable_3AssignVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/WeightsSaver/Identity_3/Identity"/device:CPU:0*
dtype0
�
Saver/ReadVariable_36ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
Saver/ReadVariable_37ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
Saver/ReadVariable_38ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
Saver/Constant_14Const"/device:CPU:0*n
valueeBcBYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
M
Saver/Constant_15Const"/device:CPU:0*
valueB
B *
dtype0
r
Saver/Restore_4	RestoreV2Saver/ConstantSaver/Constant_14Saver/Constant_15"/device:CPU:0*
dtypes
2
N
Saver/Identity_4/IdentityIdentitySaver/Restore_4"/device:CPU:0*
T0
�
Saver/AssignVariable_4AssignVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/BiasSaver/Identity_4/Identity"/device:CPU:0*
dtype0
a
Saver/ReadVariable_39ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
a
Saver/ReadVariable_40ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
a
Saver/ReadVariable_41ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
X
Saver/Constant_16Const"/device:CPU:0* 
valueBBglobal_step*
dtype0
M
Saver/Constant_17Const"/device:CPU:0*
valueB
B *
dtype0
r
Saver/Restore_5	RestoreV2Saver/ConstantSaver/Constant_16Saver/Constant_17"/device:CPU:0*
dtypes
2	
N
Saver/Identity_5/IdentityIdentitySaver/Restore_5"/device:CPU:0*
T0	
n
Saver/AssignVariable_5AssignVariableOpglobal_stepSaver/Identity_5/Identity"/device:CPU:0*
dtype0	
�
Saver/ReadVariable_42ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
Saver/ReadVariable_43ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
Saver/ReadVariable_44ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
u
Saver/Constant_18Const"/device:CPU:0*=
value4B2B(Input/Flatten/OutputLayer/Linear/Weights*
dtype0
M
Saver/Constant_19Const"/device:CPU:0*
valueB
B *
dtype0
r
Saver/Restore_6	RestoreV2Saver/ConstantSaver/Constant_18Saver/Constant_19"/device:CPU:0*
dtypes
2
N
Saver/Identity_6/IdentityIdentitySaver/Restore_6"/device:CPU:0*
T0
�
Saver/AssignVariable_6AssignVariableOp(Input/Flatten/OutputLayer/Linear/WeightsSaver/Identity_6/Identity"/device:CPU:0*
dtype0
c
Saver/ReadVariable_45ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
c
Saver/ReadVariable_46ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
c
Saver/ReadVariable_47ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
Y
Saver/Constant_20Const"/device:CPU:0*!
valueBBglobal_epoch*
dtype0
M
Saver/Constant_21Const"/device:CPU:0*
valueB
B *
dtype0
r
Saver/Restore_7	RestoreV2Saver/ConstantSaver/Constant_20Saver/Constant_21"/device:CPU:0*
dtypes
2	
N
Saver/Identity_7/IdentityIdentitySaver/Restore_7"/device:CPU:0*
T0	
o
Saver/AssignVariable_7AssignVariableOpglobal_epochSaver/Identity_7/Identity"/device:CPU:0*
dtype0	
�
Saver/GroupNoOp^Saver/AssignVariable^Saver/AssignVariable_1^Saver/AssignVariable_2^Saver/AssignVariable_3^Saver/AssignVariable_4^Saver/AssignVariable_5^Saver/AssignVariable_6^Saver/AssignVariable_7"/device:CPU:0
2
Saver/Group_1NoOp^Saver/Group"/device:CPU:0
X
ReadVariableReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
Z
ReadVariable_1ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
\
ReadVariable_2ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
Z
ReadVariable_3ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
�
UninitializedVariables/StackPackeval_step/IsInitializedglobal_step/IsInitialized3Input/Flatten/OutputLayer/Linear/Bias/IsInitializedgInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/IsInitializedNInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/IsInitializedjInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/IsInitialized6Input/Flatten/OutputLayer/Linear/Weights/IsInitializedglobal_epoch/IsInitializedKInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/IsInitialized"/device:CPU:0*
T0
*

axis *
N	
\
!UninitializedVariables/LogicalNot
LogicalNotUninitializedVariables/Stack"/device:CPU:0
�
UninitializedVariables/ConstantConst"/device:CPU:0*�
value�B�	Beval_step:0Bglobal_epoch:0B[Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias:0B*Input/Flatten/OutputLayer/Linear/Weights:0Bglobal_step:0B^Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights:0B?Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias:0BBInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights:0B'Input/Flatten/OutputLayer/Linear/Bias:0*
dtype0
h
(UninitializedVariables/BooleanMask/ShapeConst"/device:CPU:0*
valueB"	   *
dtype0
g
+UninitializedVariables/BooleanMask/ConstantConst"/device:CPU:0*
valueB
 "    *
dtype0
�
(UninitializedVariables/BooleanMask/StackPack+UninitializedVariables/BooleanMask/Constant"/device:CPU:0*

axis *
N*
T0
i
-UninitializedVariables/BooleanMask/Constant_1Const"/device:CPU:0*
valueB
 "   *
dtype0
�
*UninitializedVariables/BooleanMask/Stack_1Pack-UninitializedVariables/BooleanMask/Constant_1"/device:CPU:0*
T0*

axis *
N
i
-UninitializedVariables/BooleanMask/Constant_2Const"/device:CPU:0*
valueB
 "   *
dtype0
�
*UninitializedVariables/BooleanMask/Stack_2Pack-UninitializedVariables/BooleanMask/Constant_2"/device:CPU:0*
T0*

axis *
N
q
1UninitializedVariables/BooleanMask/OnesLike/ShapeConst"/device:CPU:0*
valueB"   *
dtype0
u
9UninitializedVariables/BooleanMask/OnesLike/Ones/ConstantConst"/device:CPU:0*
valueB
 "   *
dtype0
�
5UninitializedVariables/BooleanMask/OnesLike/Ones/FillFill1UninitializedVariables/BooleanMask/OnesLike/Shape9UninitializedVariables/BooleanMask/OnesLike/Ones/Constant"/device:CPU:0*
T0*

index_type0
�
/UninitializedVariables/BooleanMask/StridedSliceStridedSlice(UninitializedVariables/BooleanMask/Shape(UninitializedVariables/BooleanMask/Stack*UninitializedVariables/BooleanMask/Stack_15UninitializedVariables/BooleanMask/OnesLike/Ones/Fill"/device:CPU:0*
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask 
i
-UninitializedVariables/BooleanMask/Constant_3Const"/device:CPU:0*
valueB
 "    *
dtype0
�
*UninitializedVariables/BooleanMask/Stack_3Pack-UninitializedVariables/BooleanMask/Constant_3"/device:CPU:0*
T0*

axis *
N
�
'UninitializedVariables/BooleanMask/ProdProd/UninitializedVariables/BooleanMask/StridedSlice*UninitializedVariables/BooleanMask/Stack_3"/device:CPU:0*

Tidx0*
	keep_dims( *
T0
j
*UninitializedVariables/BooleanMask/Shape_1Const"/device:CPU:0*
valueB"   *
dtype0
�
*UninitializedVariables/BooleanMask/ReshapeReshape'UninitializedVariables/BooleanMask/Prod*UninitializedVariables/BooleanMask/Shape_1"/device:CPU:0*
T0*
Tshape0
i
-UninitializedVariables/BooleanMask/Constant_4Const"/device:CPU:0*
valueB
 "   *
dtype0
�
*UninitializedVariables/BooleanMask/Stack_4Pack-UninitializedVariables/BooleanMask/Constant_4"/device:CPU:0*
T0*

axis *
N
i
-UninitializedVariables/BooleanMask/Constant_5Const"/device:CPU:0*
valueB
 "����*
dtype0
�
*UninitializedVariables/BooleanMask/Stack_5Pack-UninitializedVariables/BooleanMask/Constant_5"/device:CPU:0*
T0*

axis *
N
i
-UninitializedVariables/BooleanMask/Constant_6Const"/device:CPU:0*
valueB
 "   *
dtype0
�
*UninitializedVariables/BooleanMask/Stack_6Pack-UninitializedVariables/BooleanMask/Constant_6"/device:CPU:0*
T0*

axis *
N
s
3UninitializedVariables/BooleanMask/OnesLike_1/ShapeConst"/device:CPU:0*
valueB"   *
dtype0
w
;UninitializedVariables/BooleanMask/OnesLike_1/Ones/ConstantConst"/device:CPU:0*
valueB
 "   *
dtype0
�
7UninitializedVariables/BooleanMask/OnesLike_1/Ones/FillFill3UninitializedVariables/BooleanMask/OnesLike_1/Shape;UninitializedVariables/BooleanMask/OnesLike_1/Ones/Constant"/device:CPU:0*
T0*

index_type0
�
1UninitializedVariables/BooleanMask/StridedSlice_1StridedSlice(UninitializedVariables/BooleanMask/Shape*UninitializedVariables/BooleanMask/Stack_4*UninitializedVariables/BooleanMask/Stack_57UninitializedVariables/BooleanMask/OnesLike_1/Ones/Fill"/device:CPU:0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
i
-UninitializedVariables/BooleanMask/Constant_7Const"/device:CPU:0*
valueB
 "    *
dtype0
�
.UninitializedVariables/BooleanMask/ConcatenateConcatV2*UninitializedVariables/BooleanMask/Reshape1UninitializedVariables/BooleanMask/StridedSlice_1-UninitializedVariables/BooleanMask/Constant_7"/device:CPU:0*

Tidx0*
T0*
N
�
,UninitializedVariables/BooleanMask/Reshape_1ReshapeUninitializedVariables/Constant.UninitializedVariables/BooleanMask/Concatenate"/device:CPU:0*
T0*
Tshape0
i
-UninitializedVariables/BooleanMask/Constant_8Const"/device:CPU:0*
valueB
 "����*
dtype0
�
*UninitializedVariables/BooleanMask/Stack_7Pack-UninitializedVariables/BooleanMask/Constant_8"/device:CPU:0*
T0*

axis *
N
�
,UninitializedVariables/BooleanMask/Reshape_2Reshape!UninitializedVariables/LogicalNot*UninitializedVariables/BooleanMask/Stack_7"/device:CPU:0*
T0
*
Tshape0
w
(UninitializedVariables/BooleanMask/WhereWhere,UninitializedVariables/BooleanMask/Reshape_2"/device:CPU:0*
T0

�
*UninitializedVariables/BooleanMask/SqueezeSqueeze(UninitializedVariables/BooleanMask/Where"/device:CPU:0*
squeeze_dims
*
T0	
i
-UninitializedVariables/BooleanMask/Constant_9Const"/device:CPU:0*
valueB
 "    *
dtype0
�
)UninitializedVariables/BooleanMask/GatherGatherV2,UninitializedVariables/BooleanMask/Reshape_1*UninitializedVariables/BooleanMask/Squeeze-UninitializedVariables/BooleanMask/Constant_9"/device:CPU:0*
Taxis0*
Tindices0	*
Tparams0
W
UninitializedResources/ConstantConst"/device:CPU:0*
valueB *
dtype0
5
ConstantConst*
valueB
 "    *
dtype0
�
ConcatenateConcatV2)UninitializedVariables/BooleanMask/GatherUninitializedResources/ConstantConstant*
T0*
N*

Tidx0
�
UninitializedVariables_1/StackPackglobal_step/IsInitialized3Input/Flatten/OutputLayer/Linear/Bias/IsInitializedgInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/IsInitializedNInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/IsInitializedjInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/IsInitialized6Input/Flatten/OutputLayer/Linear/Weights/IsInitializedglobal_epoch/IsInitializedKInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/IsInitialized"/device:CPU:0*

axis *
N*
T0

`
#UninitializedVariables_1/LogicalNot
LogicalNotUninitializedVariables_1/Stack"/device:CPU:0
�
!UninitializedVariables_1/ConstantConst"/device:CPU:0*�
value�B�Bglobal_epoch:0B[Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias:0B*Input/Flatten/OutputLayer/Linear/Weights:0Bglobal_step:0B^Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights:0B?Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias:0BBInput/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights:0B'Input/Flatten/OutputLayer/Linear/Bias:0*
dtype0
j
*UninitializedVariables_1/BooleanMask/ShapeConst"/device:CPU:0*
valueB"   *
dtype0
i
-UninitializedVariables_1/BooleanMask/ConstantConst"/device:CPU:0*
valueB
 "    *
dtype0
�
*UninitializedVariables_1/BooleanMask/StackPack-UninitializedVariables_1/BooleanMask/Constant"/device:CPU:0*
T0*

axis *
N
k
/UninitializedVariables_1/BooleanMask/Constant_1Const"/device:CPU:0*
valueB
 "   *
dtype0
�
,UninitializedVariables_1/BooleanMask/Stack_1Pack/UninitializedVariables_1/BooleanMask/Constant_1"/device:CPU:0*
T0*

axis *
N
k
/UninitializedVariables_1/BooleanMask/Constant_2Const"/device:CPU:0*
valueB
 "   *
dtype0
�
,UninitializedVariables_1/BooleanMask/Stack_2Pack/UninitializedVariables_1/BooleanMask/Constant_2"/device:CPU:0*
T0*

axis *
N
s
3UninitializedVariables_1/BooleanMask/OnesLike/ShapeConst"/device:CPU:0*
valueB"   *
dtype0
w
;UninitializedVariables_1/BooleanMask/OnesLike/Ones/ConstantConst"/device:CPU:0*
valueB
 "   *
dtype0
�
7UninitializedVariables_1/BooleanMask/OnesLike/Ones/FillFill3UninitializedVariables_1/BooleanMask/OnesLike/Shape;UninitializedVariables_1/BooleanMask/OnesLike/Ones/Constant"/device:CPU:0*
T0*

index_type0
�
1UninitializedVariables_1/BooleanMask/StridedSliceStridedSlice*UninitializedVariables_1/BooleanMask/Shape*UninitializedVariables_1/BooleanMask/Stack,UninitializedVariables_1/BooleanMask/Stack_17UninitializedVariables_1/BooleanMask/OnesLike/Ones/Fill"/device:CPU:0*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
k
/UninitializedVariables_1/BooleanMask/Constant_3Const"/device:CPU:0*
valueB
 "    *
dtype0
�
,UninitializedVariables_1/BooleanMask/Stack_3Pack/UninitializedVariables_1/BooleanMask/Constant_3"/device:CPU:0*
T0*

axis *
N
�
)UninitializedVariables_1/BooleanMask/ProdProd1UninitializedVariables_1/BooleanMask/StridedSlice,UninitializedVariables_1/BooleanMask/Stack_3"/device:CPU:0*

Tidx0*
	keep_dims( *
T0
l
,UninitializedVariables_1/BooleanMask/Shape_1Const"/device:CPU:0*
valueB"   *
dtype0
�
,UninitializedVariables_1/BooleanMask/ReshapeReshape)UninitializedVariables_1/BooleanMask/Prod,UninitializedVariables_1/BooleanMask/Shape_1"/device:CPU:0*
T0*
Tshape0
k
/UninitializedVariables_1/BooleanMask/Constant_4Const"/device:CPU:0*
valueB
 "   *
dtype0
�
,UninitializedVariables_1/BooleanMask/Stack_4Pack/UninitializedVariables_1/BooleanMask/Constant_4"/device:CPU:0*
T0*

axis *
N
k
/UninitializedVariables_1/BooleanMask/Constant_5Const"/device:CPU:0*
valueB
 "����*
dtype0
�
,UninitializedVariables_1/BooleanMask/Stack_5Pack/UninitializedVariables_1/BooleanMask/Constant_5"/device:CPU:0*
T0*

axis *
N
k
/UninitializedVariables_1/BooleanMask/Constant_6Const"/device:CPU:0*
valueB
 "   *
dtype0
�
,UninitializedVariables_1/BooleanMask/Stack_6Pack/UninitializedVariables_1/BooleanMask/Constant_6"/device:CPU:0*
T0*

axis *
N
u
5UninitializedVariables_1/BooleanMask/OnesLike_1/ShapeConst"/device:CPU:0*
valueB"   *
dtype0
y
=UninitializedVariables_1/BooleanMask/OnesLike_1/Ones/ConstantConst"/device:CPU:0*
valueB
 "   *
dtype0
�
9UninitializedVariables_1/BooleanMask/OnesLike_1/Ones/FillFill5UninitializedVariables_1/BooleanMask/OnesLike_1/Shape=UninitializedVariables_1/BooleanMask/OnesLike_1/Ones/Constant"/device:CPU:0*
T0*

index_type0
�
3UninitializedVariables_1/BooleanMask/StridedSlice_1StridedSlice*UninitializedVariables_1/BooleanMask/Shape,UninitializedVariables_1/BooleanMask/Stack_4,UninitializedVariables_1/BooleanMask/Stack_59UninitializedVariables_1/BooleanMask/OnesLike_1/Ones/Fill"/device:CPU:0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0
k
/UninitializedVariables_1/BooleanMask/Constant_7Const"/device:CPU:0*
valueB
 "    *
dtype0
�
0UninitializedVariables_1/BooleanMask/ConcatenateConcatV2,UninitializedVariables_1/BooleanMask/Reshape3UninitializedVariables_1/BooleanMask/StridedSlice_1/UninitializedVariables_1/BooleanMask/Constant_7"/device:CPU:0*

Tidx0*
T0*
N
�
.UninitializedVariables_1/BooleanMask/Reshape_1Reshape!UninitializedVariables_1/Constant0UninitializedVariables_1/BooleanMask/Concatenate"/device:CPU:0*
T0*
Tshape0
k
/UninitializedVariables_1/BooleanMask/Constant_8Const"/device:CPU:0*
valueB
 "����*
dtype0
�
,UninitializedVariables_1/BooleanMask/Stack_7Pack/UninitializedVariables_1/BooleanMask/Constant_8"/device:CPU:0*
T0*

axis *
N
�
.UninitializedVariables_1/BooleanMask/Reshape_2Reshape#UninitializedVariables_1/LogicalNot,UninitializedVariables_1/BooleanMask/Stack_7"/device:CPU:0*
T0
*
Tshape0
{
*UninitializedVariables_1/BooleanMask/WhereWhere.UninitializedVariables_1/BooleanMask/Reshape_2"/device:CPU:0*
T0

�
,UninitializedVariables_1/BooleanMask/SqueezeSqueeze*UninitializedVariables_1/BooleanMask/Where"/device:CPU:0*
squeeze_dims
*
T0	
k
/UninitializedVariables_1/BooleanMask/Constant_9Const"/device:CPU:0*
valueB
 "    *
dtype0
�
+UninitializedVariables_1/BooleanMask/GatherGatherV2.UninitializedVariables_1/BooleanMask/Reshape_1,UninitializedVariables_1/BooleanMask/Squeeze/UninitializedVariables_1/BooleanMask/Constant_9"/device:CPU:0*
Taxis0*
Tindices0	*
Tparams0
�
Initializers/Variables/GlobalNoOpo^Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/InitializationAssignr^Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/InitializationAssignS^Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/InitializationAssignV^Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/InitializationAssign;^Input/Flatten/OutputLayer/Linear/Bias/InitializationAssign>^Input/Flatten/OutputLayer/Linear/Weights/InitializationAssign"^global_epoch/InitializationAssign!^global_step/InitializationAssign
%
Initializers/Resources/SharedNoOp
M
GroupNoOp^Initializers/Resources/Shared^Initializers/Variables/Global
E
Initializers/Variables/LocalNoOp^eval_step/InitializationAssign
$
Initializers/Resources/LocalNoOp
"
Initializers/Lookup/TablesNoOp
j
Group_1NoOp^Initializers/Lookup/Tables^Initializers/Resources/Local^Initializers/Variables/Local
��
%train_dataset/TensorToOutput/ConstantConst*��
value�B�	�	"Էk��?��n N?�٣��y�?�}�?ɓ?O%F?�%m���'?���>����NY�?�����ϛ?��+>E���d|�dAt?@��>m�G�uyE�l��>벪��ޗ>���}�!����f�!V��쮾0Г?Y1�?@��?�-?���>��6�@�E���d|�O%F?�%m��Ȼ��ξ�@3
۾W�?��[�:o��˝��E��g��==֗����4	�cZ����%?l��w4˾��+>E��bm?J�>=֗�u�_��ĝ��G?�C�?�$��٣�h~¾E��*�M�����'����ϾMF���G?��o���?N�G��$�@a>�>������?¬�>��>ʏ>�@< ����:o��E���U�`���|���ͫ�uyE���1����?�-�K��܍>lk��bm?����Z���w뿽�'��)*>o,���?N�G�xB>E���Q]��C+��@;���W���t�)P=?��_?�$���x���E����?c����ӯ�_3_�a�?z;��>h<U?J��E���d|� ��?E���W�y>����a:@�$��O4��]Q����?�Q�?Hh�϶1@u�_�(){?�տ Ý>Y;y�K����ԡ�>"��gÀ�7ٛ?w뿽_3_�����4���3?�$�������R�?2��>_�v��\6>�͛?c�c?cZ����N�&?�$��T'F��������ڎ׿3��=7ʏ>ʇ@a�?U�5���?%�M���r�*�M��?�4����w�l�p=ڄ��/��>�7_��O4�͓\�D���y+�Sv��wm�?��g��j�?k����>Y;y��S�?��r�!���M�><�>��w�	�����1���B?�����$��T'F�r��ى�@S��x>��<��@��>�h�<W�?K���y�?r�A�&��m�>Z���hݏ���t��������̢��z� �f��!���"��%K8�@��>V�7�l�p=����Զ����{>g�@�:�r�!-@/
��|��(��(){?(.��xua����K��v������>�=�M�>*�?u�_��?�����%?JJ����x��h����?	(@5u�Z���У�MF����> Ý>���XD_>:o���q(��d�?��?z[	�����-��C��w���ޗ>%�M�������>�m��]�P���x>��7?�j�?)*>l�R����>�W�>�S��F�?M�@\�>�@;�(����t�1����̢��#Xo�����Q>A���uG��=t@u�_�ʇ@@��?	�οW�?�$��:o�������'�?c���hݏ��4	�B���=����?%�M��.>0iG?2��>�u?���7ʏ>���>^�N���G>�!��N�G�����?�r��M��˴?���>k����_?�-���	�Q�D���B�h�]V?7ٛ?7ʏ>MF���{�Y��?�!���G*�������?��;=]�P��z�?(����?�]�?Զ��K�?XD_>f��E���>>����?�쮾x�?�R4?�ۚ��W?��G>z� �;j<lk��9
?g��=�+u?�W?�'��^�N��J�=����W�>�h��}Y��f� ��?��@ ��?��t��߰�[:=��28��$��ۺ/��M7���Q}ӿj��?���=
��?�C�r�>�#������=�q(��t�p�X���x>��O?���=����G>ȢQ�	�Q����?۲>�+?3��=V�7��?�2�"���������@�@=����Q]��O[=���>��>�R4?���U�5�1��?'}�?	�g���?]+�?t�]�]��?��<S�:@�pI@�3N�[�@�W�>��=r��t�wd|?�O���?���?FwоI}'��ޗ>%�M�	�Q�r��y6�k���|�����R4?P6�H�?��ན�-?�q~��R�?Q��?U���G�K��Ȼ�D������ƀ>JJ��N�G���;���a����d��a�}�V�7��?���R=��o���{>��x��S����?�|?��7��|�����>O?���o,���@?�G*�����@�1䃾�z>`�?hݏ�l�p=P6���#[?���S����?Y'?�xj�@��>��<�R4?�ۚ�3
۾�M���O4���;˝��t�B��� "?(��D���>����������c�>!���A:�=k��<�>?�'�D���¦]�I}'���G>���@f��Sux?
9��K?<�>w뿽9Ӟ��,@��W�?'}�?�R�r� <~<E�5@�|����>��@�Y�?�4��#[?�G*��$�]?�r�?2=E=���=��-����/��>�!������+>A�_?���?��k���x>V�7�9Ӟ������h�<�ޗ>�$���$�0iG?/b=@��̽�|���ӯ�:]�TXi?�W?�$��$���h�r��Ѿ�M�>�������t�V��?��o���?��x�v����q(��y6��B�z[	���'?���?��> Ý>���w4˾���>��@�aF�/� ?Z������>uyE�^�N��OQ?b����٣��$���@��Ѿ�h?B�@f��>uyE�FwоI}'�1+��W�>�����>��ؽ��7�2=E=��l�p=^�N�K��������[�h~¾���?1䃾����2=E=��c�c?ڄ����o�QԾ��?:o��lk��B�h�5u�P��?��W���@k����B?������:o��!���"�I�'�?*�?(����t��{�K���9l���x�}��q(��Q]��⛿Ӥ2?�_>��K?2�"��u?�28�w4˾�q~�D���*�M��q-�E�V�7�9Ӟ�r��@�-?W�?r��?��+>����y6��\Y@<#�x�?�?�]�?�w����?��h?���?r�J�l�\�>G�K�7ʏ>��K?�?�h�<W�?G�?w�+?����e����?W�ɿ(��Y1�?)P=?0�?�4E� ��>J���F�?
�U?.��?�|��w뿽�4	���p������?�����@���=�Ǔ>g��=�⿢�)@D�������"�?����%�M��S�D���	�.�_O(?<�>�қ�uyE�Fwо����d/�XD_>
��>��?B�M?�(��%m���_3_�¦]�������w4˾�q~���?�w�>�$C��\6>�ӯ��4	������3N������G*���ؾlk���ѾJz�`�?��Ͼl��?(.���ƀ>̢����!�����ى�\�>�+u?��>:]���1��w���l.=w4˾J��Չ�?Y'?!&�=�d?p׋���t����z;����>�J0@,H�����=���?'�?��x>�g?l�p=2�"�`yD�#[?��=ZW�]?��;=�s��|��0Г?(){?�@������?��=�R����>9
?� �?�Ȱ�hݏ�l�p=P6�`yD��>K��v������?���?��i>�� ?hݏ�	����ۚ�K����?%�M��h�r��d|�=!�*�?�_>O?��K?h��N�&?N�G��۠��M7�aF����c�����g���>¦]���%?�!���G*�:o���M7�"����.�@��>���4	��C��OQ?< ��h<U?UX�>r�*�M��z>ma?��9Ӟ�vׄ�`yD��X�]B�@r�K����?��?]�P�:�S??�'��j�?�G?ؒ;?��q�w4˾xB>E��ဲ�G�>'0о�Ȼ�uyE����o,�96R�N�G��,;�g��B�h�/�=Ӥ2?��uyE�2�"�"�?̢��%�M��q~�)N<�{��z>:�S?��W��'��ʏ>o,��>��>��0iG?�@O%F?�쮾hݏ�:]�ʏ>G��>QԾ��?�.>�F�?!�.?gd?-����w��ξvׄ� Ý>������x�xB>˝�������j��?u�_�(){?2�"� Ý>�!���G*��h��}Y���Ǜ���x>��W��ξ����0���>�G*��]Q����ŀ��'D^�ma?w뿽O?)P=?0�?�4E� ��>J���F�?
�U?.��?�|��w뿽�4	�ʏ>�0����?w4˾�܍>ԡ�>�f��q-��4���_>�R4?lf@z;�W�?��x��@=E���d|����?G�K��ͫ�uyE�����r�>����z� �h~¾�}Y��e������쮾w뿽��t��?��o����?��x�=8�?�}Y��Q]�wd|?���!@��t���?�����h??@�۠�E�����&��>z[	�Ճ?�r�?��1��0�����?�٣�v������=A�&�<>?<�>���>�>cZ����?�7_��٣��۠�A�_?2��>�B�E���?��>%Tl�s�&��?�٣����0iG?aC+>�s��\6>f��>�R4?^�N�o,������=�S�����m��q=�+u?��w�O?k���OQ?�4E��O4��B
@E��aF�lȾ�� ?V�7�:]�^�N�o,������=�S�����m��q=�+u?��w�O?¦]���>�7_�z� �}��R�?�VJ>B��� "?���=��-��{���G>1+��$�����!���!������Ӥ2?u�_�Y1�?B����C�?d/����@=���aC+>/�=z[	���7?:]��߰�G��>Y;y��٣�,H��)N<	�.��ɿP��?��7?�?��d>�~�?�F���G*�����q(��{��n��Ȱ�?�'���t�2�"�xua���{>w4˾&�lk�����A��z[	�¬�>��K?�{����?�M�����@�۠���?�|?]V?E�V�7�D�����p��~�?����w4˾f���q(��>>�N㔾3��=��>��>)*>3
۾��G>��=�����>A:�=O%F?���> ��?MF��B����h�<������[�h~¾r��d|��h۾'0оw뿽�ξa�?[:=��u?G�?�.>˝�����KH=�Z���w뿽E9�?B���*ߩ?&D�=r��?:o��D���J�l�l���d?��>(){?�������̢��z� �f��!���"��%K8�@��>V�7�l�p=��? Ý>�M���W�>�S�lk���@�+?z[	���Ͼ:]�ad@z;���?��[��۠��.?�{��+?<#���>l�p=���=����?��[�-��@)N<
9��x>���\Ո@D���k��z;���?w4˾�����>B�M?\�>@��>���ĝ�2�"���B?�ޗ>%�M���ؾ�R�?���?�:��Z���ԽG?ʇ@l��>xua���{>�٣��������t�X���|���g?�R4?��?1�
�W�?�$�����.?� @&��>z[	�w뿽:]��m�?I}'���?XD_>xB>A�_?���?�f�?=֗�m�G��ξ���=��懡?K���m=�q(��y6��u�@��>���=�4	�V��?`yD��u?��x�:o���M7��f���;0ҿ0Г?c�c?2�"��q~���q���X�½��?�~]?/�=ma?��:]��K@�Ln?����K���i?˝�����q�M���x>�ӯ���-�Fwоz;�&D�=�G*���E���d|��i�2=E=u�_�(){?P6� Ý>�-�w4˾�7?˝�)c����-�����R4?I�?,h���?��?���>����y6�-�?��X�F��?9Ӟ�ʏ>��W�? ��>�R����=��?O%F?E��ӯ��?�I�E@C> ���@ ��>�R����=��ؽ�@0ҿ�g?�ξ�$�>��?]��>z� ��,v@E��R�>��C>���h�-@�'��wh@�4��}�@��x����r�!�� ��?��3�Ճ?�?�����z;�������[��*�lk��y+��,��@��>У�:]���1�$��?�#����x����g�Q]�n��� ?��˿ʇ@cZ���|?�X�w4˾q��>�M7���?j.�>G�K�7ʏ>��t����ƀ>JJ��N�G���;���a����d��a�}�V�7��?�2�"��S4?̢���G*��.>E��"��~��?՛@�g?��-�Fwо��_?̢��#Xo�����q(�*�M�l���%m���)@D����?`yD���?��w�+?r��d|�O%F?�|����?�?�2�"�xua���{>w4˾&�lk�����A��z[	�¬�>��K?�]�?�o��|��?K��ZW�r�����D�>3��=�ȫ?��>)*>��l���G*��U5��M7���!V���\��g?�R4?�K@�1�
����>�O4��@=�M7��m�������\6>(��D�����1��q~�&D�=%�M�xB>ԡ�>�VJ>�,��Ӥ2?}% �O?cZ���񱿻��?�W�>���>����>>���P?�� ?��>�>¦]���>�7_�z� �}��R�?�VJ>B��� "?���=��-���>��B?�+��W�>�h�E�����&��>E�?�'��4	��C���B?����w4˾f��r��y6��S?�R�?���?MF��^�N��g?����%�M����>]?�m�������+u?���>�m�? Ý>ۓ�?��x�v������ <~<��	@'0о��	���)*>�-?���w4˾X�½��@���j]??��X��ӯ��ξ�{�z;�������&����>��6Eڿ:�S??�'���?@��?U�5�N�&?%�M���ؾ@�ii> G�><�>Gˣ?l�p=�ۚ�����1+�K��ۺ/�lk��q
¾Fo���7�?��(){?^�N��3N��7_��G*�v���D���Y ����Ӥ2?��_3_�I�?�q~�W�?)4?h~¾����m��px�?2=E= ��?O?cZ����>����%�M���=���>۲>���?7ٛ?}% ��'���!@/��>W�?XD_>:o��E���U�cq�?c�����c�c?����k��?������x�f�����"��B ½'0о��w��?��������������٣������@�ii>��k���?��?y>��p��4����{>K��ZW��M7�Q���#"ҿ "?(���r�?�x?��>��G>��x���m=Sux?R�>cq�?��\���_3_��G?z;����>��f��r��e�8R#>Z�����7?���>�@l�R����?��=�S�Չ�?y0F?-�?�Ȱ�¬�>l�p=^�N�z;����N�G��R����>�=B�����x>ԽG?:]��K@���?Y;y����q~���@�	�.�'�?�7�?m�G�	���a�?�s��Q��?XD_>�$�r��e���7��4�����?Y1�?�{�B��?�!�����@�,;�r�5����n��z�?��w�E9�?vׄ��w��l��XD_>!���@�A���Y�Ŀ��*�(�����?l��>�񱿍u?z�@�i?˝�a���f��?��\��ӯ���>��?I}'�1��?%�M��U5�D���A�&�]�P��\6>�W?�)@��ƿ��B?������x���˝��ى��ȿ�=%@��ʇ@��R=�J?��?w4˾�3
?!���9o�?\�>a�}�V�7��ĝ�(.���ƀ>1+�S"�?h~¾�}�?B�M?�VԼ2=E=p׋�D���k����o��>��:o��Q>1䃾-0�<�>^Ƴ?S�:@^�N�r�>d/���x��q~����>"�I��L�� "?��Ͼ9Ӟ�lf@�0��W�?���>�,;����?�r�~��?�@;��g?_3_�T6��벪���q�z� �}�˝��ى�$�1��� ?V�7�"�4@��R=�OQ?QԾ��x���=g���~]?�t�>�@;���ϾD���^�N�U�5��$�%�M�X�½�F�?
9�����쮾���ξcZ����?�7_��٣��۠�A�_?2��>�B�E���?��>��p�Ln?JJ���٣��$���?��7@�,��'0о��g�MF��>2����������G*�,H���.?۲>��]�Ӥ2?F��?�>�$�>z;�JJ����x�;j<���>B�h�Ds�=3��=�ҋ?�?TXi?�J�=��q�w4˾��?�}Y�2��>�q>>2����@_3_�B����3N����>����<@r�J�l��+?��\��ӯ�:]�P6�=���+�w4˾�۠�r��Q]��z>E�hݏ�D���Ţ.?��>#[?�G*�,H�����>Y'?��?�+u?��:]�cZ����G>�X���-?	�Q��M7��d�?&��>��X�u�_���t�)*>o,���?N�G�xB>E���Q]��C+��@;���W���t���>NY�?1+���=�@=�q(��y6��Da?�� ?���?�l��>��%?�X����>��+>!���1䃾���?��X���Ͼ��-��$�>��?�T�<�|?�}�?Ƅ]@aS?@wd|?a�}�^Ƴ?D���ʏ>=��JJ��%�M�(��=��?}>�?�6J?G�K���W�D���¦]��3N��F���G*�,H��!���	�.����� "?�ҋ?��K?����W?l��w4˾X�½���=��?`���쮾���?���d>��X?�7_��O4���m=����ى����>���}% ��GQ�P6��OQ?�>w4˾bN���@R�>B����쮾(��_3_�¦]�/��>�T�K��,;�Q>�y�?/�=ݰ�?���'���C�a�?Y;y�z� ��.>���
9�X���s񾲸w��?����=xua���{>��ZW�Sux?A0>迓�<�>���>���?�x?�4����?��=�R��M7��Ѿ e�?@��>�g?l�p=2�"�l�R��>���۠�Չ�?�6?q=Ӥ2?��Ͼ:]�Ţ.?xua�懡?w4˾;j<D���A0>s)��E����?>2����N�&?��[�ZW�T@J+f@��;�'0о��Ͼά?������G>�l.=z� �@��@)N<�Ǔ>/�=�w���y@�'���n>@�h�<z�(@���?�7?�.?bm?n�+@��\���7?,V@�C��u?����%�M�:o��˝�����+?]��?��W�D�����1�/��>������v���)N<q
¾�z>��?7ʏ>MF���?��o���?��x�=8�?r��Q]�wd|?��Q�@��t������4����?h<U?!�Q>�{�-`)��� ?��<O?2�"��ƀ>< ���G*��S�)N<Q���~�<�>�ӯ�L� �¦]��W?�4E�<�|?q��>)N<�l@g��=��X�hݏ�:]�k���|?����K���q~�g��q
¾'�?*�?(����t�@��?I}'�n N?��x�
��>g���m�����>a�}����=���>��>U�5�n N?�$����ؾ��?�r�;a���@;��ӯ�E9�?�a�?U�5�|��?�٣�KS?lk���=�+?����w?��-�P6���>< ��w4˾�@=D�����4��>@��>hݏ��ξ�@�q~�W�?��=�,;��}�?!�.?cq�?'����?�4	���>U�5��3?w4˾��$�lk��"�I��:��-���ӯ�E9�?(.����N�&?�$��,H����@��U��n�P��?¬�>�>�]�?G��>W�?�٣���@E��J�l�cq�?a�}���Xi�{�z;�����٣���ؾr��d|��(��� ?w뿽��t�2�"���B?�!���G*�	�Q�)N<�y�?q�>�˴?����-��ۚ���%?�9l�%�M��S�lk��y+�%K8�*�?�����?��@�4E���x�5��?�}Y��y6���7���x>}% �_3_��?z;��h?��=�q1?�R�?Y'?9>z[	����>��>��>l�R��3?�$��c�>r���������=֗���ά?��1���>�#���G*��$��}�?��ؽ�(�Ӥ2?��y>wh@`yD�W�?%�M��$���@�*�M���?W�ɿՃ?L� ���?�4�����>��x��,;�g��"�𾼼��3��=Gˣ?���?¦]���o���{>��!����ŀ��Ӌ�@��>?�'�ʇ@�]�?o,��h?w4˾�$�r�����?-����'?c�c?k��G��>�#��w4˾v���!���1䃾� �>�7�?}% ��?�^�N��~�?�T$��%Z@G~���;=�ߧ�ma?�ӯ�L� ���d>Ln?������?f��!����Ѿ��?@��>У���t�����?�7_���=f���q(�y+�&��>�� ?�ك���-�cZ��U�5���G>K��J����@�	�.�lȾ�s��_>�>�C�"�?�$�K�򾑎�>Q>���?���쮾��Ͼ�'����p�Զ���X�z� ��h�Q>
9�(���E�}% ��'����K?Ln?��{>%�M�;j<]?��?�u?E�u�_�9Ӟ���d>6]�?�?z� ����@D���ɓ?�	�������M@D���ʏ>�s����?�G*�bN����>y0F?@�1��� ?���>���?��1��?�?d/���UX�>ԡ�>���?�h۾G�K���w�:]��{��|?Y;y���="�>�M7��f���@�:�S?��_3_������?��@?K��(��=��@��y�?���z[	���Ͼ:]�Ţ.?6]�?�F��K��h�r���?�쮾(���>�{�z;��#��N�G�bN����=y+�]�P�ma?���?�¦]�z;�96R�N�G�:o��)N<�=��.�@��>�W?:]�����/��>��q�w4˾�$�Չ�?�"$@;�����*���W�MF��cZ��z9@�M����[����}Y�"��	���w���қ��ξ��1�G��>̢��%�M��S����=\�?���>�7�?�ӯ���-�)�$@��?ۓ�?�!@X�½�}Y��{�8'9@W�ɿ¬�>�ξ��1�Ln?����٣���ؾ���>�f��N>:�S?���=�?�>��w���������]Q��}�?qE ?1;.�B�@�W?^R@��K?r�>�+� ��>��ԡ�>�VJ>U@�?c���hݏ���t�k��?U�5��3?��x�f���}Y��e��Da?}j�����=�'���K@��g?����K��q��>�q(��>>��%���\6>�ӯ���-���p�z9@�!������$����ŀ���z>@��>O���ĝ�>2����?�7_�XD_>�S�!������h���?��<_3_�k����%?�+�x�@/�?Ƅ]@e�?�t�>��X���w�:]�����z;��4E�����?r�aF�lȾ��X��ȫ?�?����=,h���?���,H��(�2@}>�?�GK�3��=��'?���?�?�4��W�?��x�}�Q>"�I�&��>c�����g��?���p�1�
�JJ���O4�"�>����m�������%m� ��?:]����=����{>��>!��}Y�A�&�*�x��s�ͫ��4	�U#�o,�d/��٣��:���@��?Hh⿜=t@(��]=@3�?z;�XM@�@�܍>E���e�-�?-��u�_��R4?%Tl��OQ?̢��K��D��Sux?�{��z��� ?Ճ?��>ʏ>	�ο���>z� ��U5�!����f�tʽ�c�����Ͼ��K?��p� Ý>�X��G*��q~��.?Ռ�?�Z�=��*���:]���1�lݿW�?�����@��Q]���߿�� ?(��
��?�m�?�h�<ۓ�?_?ҧ<?r��d|�'�?
]ڿx�?(){?a�?��>1+�w4˾f��A�_?��;=��8?c���(����-�����벪����>��x�͓\����?�VJ>��k�Z���:�@
��?��p�1�
�JJ���O4�"�>����m�������%m� ��?:]�I�?3
۾1��?�W�>	�Q����> <~<px�?�쮾��5@O?B����o��]��>��[��q~���@��y6��M�>�쮾7ʏ>uyE���1���>�#��%�M��h�ԡ�>Q��?#��>wm�?��Ͼ�?�>2���w�������f�������1S��@��>�ͫ�9Ӟ���1���G>1+��O4�	�Q��}Y��Q]��޾Z���u�_��'��B����w��N�&?�$������0iG?��;=H�"��\6>Gˣ?O?�]�?I}'�v
�?��?�q~�]?�ii>-�?�4���û?MF����1�1�
��+���J�ｲ}�?A0>�gX�]��?w뿽�4	��{��e@������J��E��J�l�~�Կ7ٛ?�ͫ�ά?l��>lݿ�$��٣�	�Q�1Q�����Q�? "?�ȫ?L� ���?1�
�#[?�$��s�1@E��5���cq�?G�K���:]�����*ߩ?����G*�(��=lk���ii>\�>�쮾}% ��?��K@�Ln?���%�M�6ң=lk��aF�/
�<�>�ӯ���K?�K@���o�JJ���$����0iG?!�.?���ma?m�G�9Ӟ�TXi?�3N���?��A?�.>�q(���ؽU@�?E���	���P6����?�$�XD_>X�½���R�>q=��x>(���ξ��?���X���=6ң=Q>bm?�<�?�4����g���t�¦]���%?������>�G?lk���Ѿ%K8�:�S?����K?(.��z9@�����G*�v�?r�5���ѩ&�P��?��g����>��?�h�<|��?%�M�}�r�A�&����?'����Ͼ�'��Fwо벪�W�?��>(��=A�_?2��>��C>���hݏ�9Ӟ���p��w����@?w4˾����@��U��q-���x>�ӯ�c�c?vׄ�G��>������x���+>��D@�d�>�94��R�?7ʏ>(){?¦]����?�����W�>'I�>��@��y6�l���d?(��y>�K@��z�?����%�M���ؾ�}Y�*�M�g��=�7�?(��uyE�P6�Զ����?%�M�6ң=Q>]+�?&��>���>�g?uyE�TXi?�J�=�h?�G*�,H��)N<"�I���?�� ?u�_���t��C�~�h�W�?%�M�;j<�@�)�?����@��>ԽG?O?ڄ�������28��G*��q~���?"�I��x�Ӥ2?��?O?vׄ���1+��$���i?�q(��m��!��>hs@}% ��?�lf@z;�W�?��x��@=E���d|����?G�K��ͫ�uyE������i @������=j�n>A�_?�X?/�=�@;���>�?��Y�?�0�����?�G*������}Y�y+�e�A��%m�m�G�l��?^�N�`yD��$��٣�'I�>���)c��_���:�S?��c�c?l��>�w����@?����$��q(��Q]��%����\�?�'�c�c?l��>�~�?����O4�	�Q�����d|��+?E�V�7���-�cZ��������@?�G*�j�n>˝�a���ްc�Z���?�'���?TXi?�J�=�h?�G*�,H��)N<"�I���?�� ?u�_���t�k��?�o���h?)4?bN�ԡ�>�=�z>=֗�V�7�uyE�TXi?�q~�v
�?�٣�&�E���d|���C>�쮾�W?�>��ƿ_�?< ����[�(em�D���}>�?��P��?Ճ?�n@�{����?�9l���@Y;&?�q(��e���̽��x>�ك����>T6��_�/@ۓ�?w4˾�lk��}>�?C�ֿhs@�ӯ�(){?B����4����?�$��	�Q����=��ؽ�⛿�����7?l��?P6�l�R�b�����	�Q�D����=(������>��W�MF�����?`yD����?��>��+>���>���?���?=֗���g�:]��K@��3N�W�?-�@bN�Sux?��@�M�>�\6>��'?l�p=@��?�w��|��?�٣�T'F�E��a����d���@;��_>�j�?����?���<�|?f��A�_?
9�'�?�\6>m�G�:]�TXi?�q~��3?�G*�X�½g���w�>O%F?��*���w�:]�>2����������G*�,H���.?۲>��]�Ӥ2?F��?�>���7�?������=�S�r��d|��m�>�|����W��'���K@�l�R�W�?%�M��sX>��?Q��?�,���� ?��7?c�c?�$�>U�5���{>-�@����� <~<��?<�>��7?l�p=^�N��W?̢��K��	�Q�Q>Y ��N>ma?��?�?���R=��?���>w4˾�@=lk��a>�>���>������4	�B�����?�4E���=(��=D�����cq�?*�?���>�ξ2�"���?b�����>�h�lk��Q��?#��>*�?V�7�:]�����3N�QԾK���q~����A���k���� ?V�7���-�2�"��u?�28�w4˾�q~�D���*�M��q-�E�V�7�9Ӟ������ƀ>�9l��G*������R�?
9��1���+u?x�?��@�K@�Ln?���%�M�6ң=lk��aF�/
�<�>�ӯ���K?k��?z;��u?%�M��۠�E���f����?
]ڿw뿽:]����r�>�$�w4˾	�Q�������S?�d?ԽG?:]�)*>�3N��?w4˾L��@)N<R�>�u?��꿖�M@	���k���h�<�4E�K�򾂨�?���5����x>�쮾Gˣ?D����]�?	�ο�u?��:o��r�aF�&��>'0о �@���>��R=��G>Y;y��٣����>)N<A���B���=֗���W�O?)P=?l�R���@?K��,H��E����������4����?l�p=Fwо��96R���=�,;�D����=/�=��*����ξ@��?Զ���u?K��S�r�a���*�x��Ȱ�p׋��4	�2�"��ǿ]��>%�M�}�g���*��F���z�?�g?
��?2�"���+@�����W�>'I�>�}Y�J�l�eFC?�қ��ξ�$�>z;�JJ����x�;j<���>B�h�Ds�=3��=�ҋ?�?(.��Ln?����/y*��U5��}Y�5�����-���?m�G��n@������%?����K������}�?a>�>�,�������O?V��?����&��?�$��UX�>r��t�j.�>�����<���?��1��S4?�T���h~¾!���"�I�\�>:�S?�ӯ��ĝ�ʏ>�s����?�G*�bN����>y0F?@�1��� ?���>���?)*>����?�$��v����}Y�!���u?}j���қ�:]�TXi?�h�<+�?�$����=�F�?aC+>��C>G�K���'?�b�?2�"��J�=l����=f��D���*�t?��7��� ?f��>O?B�����?�F����=��r��U�O%F?�쮾hݏ��'��^�N��J�=����W�>�h��}Y��f� ��?��@ ��?��t��G?Զ���h?N�G��*����>�m��L�n���\��û?�?%Tl���?JJ��K��,;���@�R�>\�>�z�?���'���{�G��>�����O4���;���ŀ��Fo��3��=o���4	�P6�z;���?��=J��D����r���n?�FC?u�_�y>>2���h�<Y;y�K��h�0iG?
9���E���?��>L� �2�"� Ý>���?�٣��i?˝�a�����t>3��=��L� ��{� Ý>̢��w4˾&��@�VJ>>Z˿��x>��'?�b�?ʏ>�|?����z� ���;��@��U�>���m�G��ξ��K?��B?�>��=�.>D���}>�?�%?�쮾��<9Ӟ�cZ���񱿻��?�W�>���>����>>���P?�� ?��>�>���1�
��>��x�&�r�5���֛r�:�S?���>(){?��1���>�#���G*��$��}�?��ؽ�(�Ӥ2?��y>2�"��OQ?������x���ؾ���	�.�5u�Ӥ2?���>�?���=��l��K�򾟄$�Q>1䃾�jS�a�}��͛?�?l��>z;��>K�򾖍����@��e��?2=E=f��>l�p=TXi?a�?QԾ�!@��?��?�Ce?�f�?c���?�'���t�¦]���G>�!��z� �f��Q>ဲ�/�=:�S?���ξk��벪���?z� �&��q(�Y �/�= "?m�G�:]�¦]��S4?̢���٣�	�Q���@��t�L�n���x>u�_��4	��@/��>W�?'}�?�܍>E���d|��w@'0оx�?ά?��d>벪�懡?�O4���?D����m��g��='0о��:]��K@�1�
��>w4˾�R�r�B�h��V޿3��=�қ�ά?�Y�?1�
�N�&?��=(��=!���aC+>�!Y?-��w뿽�ξk��?벪���@?�O4�f�����>�f���.���X�¬�>ά?�����w��]��>w4˾J��0iG?۲>��7��\6>��Ͼ��t�P6�/��>�F���٣��$�A�_?"G�?s)���|���ك�MF���]�?K���@	nF@v�����@�Ce?�Q�?�%m��͛?���?��ƿ@�����٣��聿���=*�t?�V޿B�@¬�>c�c?�G?`yD�N�&?��=�۠����y+��Da?�\6>��?_3_��@ Ý>ۓ�?%�M����r��m����?
]ڿ(���ξ�C�~�h�W�?%�M�;j<�@�)�?����@��>ԽG?O?r��@�g?2�@��h?�܍>r���ؽ�[@�����>�R4?Fwо1�
�&D�=�٣�:o�����>ɓ?=Do>ma?w뿽:]���R=� �?96R�K���m=)N<qE ?l��G�K�Ճ?uyE�Fwо1�
�&D�=�٣�:o�����>ɓ?=Do>ma?w뿽:]���R=��G>���w4˾J������>>�&��>-���ͫ�uyE�¦]����?�����$�����>�M7�Y ������z�??�'�y>�C�3
۾���><t&@v���D���\�?'�?3��=�ҋ?l�p=TXi?�W?�$��$���h�r��Ѿ�M�>�������t��߰�r�>�7_���x�&��@A0>ˍ[��z�?��'?�R4?��?��o��>�O4�&�E��5������@;����=l�p=������?< ���G*��,;�r��Q]�4��>E����'��¦]���>b����_+@bN�g�Q]��,�� "?��Ͼ���>��1�/��>������v���)N<q
¾�z>��?7ʏ>MF��Fwо�3N��ޗ>K��v����}�?Q��?�+.> "?��	���)*>�W?�+� ��>q��>���>M�@�+?�����W��'���ۚ��ƀ>�7_��٣�͓\�����ى��,��:�S?��'?���>�{���?̢���$��&�lk���y6��`~���?�ӯ�O?cZ���w����?%�M��$�D����Q�>'�?�|���ԓ�	����Z?�ǿW�?��>;j<lk���ѾO%F?Ӥ2?u��?�ξ�����h�<�F����x�	�Q���?� �?�<�<�쮾�қ�uyE�������>�T��G*��q~����=�ii>�m�>�FC?��Ͼ�ξ%Tl��|?Y;y���r?!���@�J�l��y濇FC?��W��r�?ad@l�R�Q��?�٣���=r�	�.��E�?c���f��>��>���?��1��?��>�BB?����e��@=֗�7ʏ>_3_�@��?��o�&��? ��>�BB?����*��+?z[	���<l��?�x?������?��x��h�˝��t�cq�?�@;�ԽG?uyE���1���o��>�O4��$����?�6?�L�=:�S?��w���t��տa�?�9l�z� ��@=)N<y+��]��՛@(���b�?@��?=��+�?��-?h~¾���������C>����uyE�wh@I}'����?XD_>c�>���>�*��ó?��*���'?��K?l��>3
۾���>�$���$��.?*�t? ��?<�>u�_�:]���K?Ln?��{>%�M�;j<]?��?�u?E�u�_�9Ӟ���d>����G>%�M�i�X?0iG?9
?#��>�Ȱ�x�?uyE��K@�1�
��>w4˾�R�r�B�h��V޿3��=�қ�ά?ʏ>G��>QԾ��?�.>�F�?!�.?gd?-����w��ξ%Tl�I}'�n N?XD_>�q~�]?�ii>/
�Ӥ2?��I@�>����l�R�|��?��	<�?E���e�&��>�|��Q�@��t���d>I}'��3?L65@	�Q�A�_?�{��$C��s��7?�r�?����ƀ>̢����=�h�g��ဲ����@��>u�_����>�$�>�4�����>��=ۺ/����B�h�پ�ma?�_>���?lf@`yD�K�?�G*�6ң=r��Q]�U@�?a�}��ك�:]������-?����w4˾J����@�J�l���7�<�>��Ͼ�?���p��B�=b�����?���}�?�w�>l��@��>���=y>�����-?������@��E���Q]�wd|?ma??�'�uyE��]�?o,��h?w4˾�$�r�����?-����'?c�c?���e@�M��w4˾�.b�lk��2��>�(�@��>�ͫ���>�i�?xua�1��?����m=ԡ�> <~<��>�4����l�p=������G>QԾ�����>D��� <~<��7�G�K���?:]�^�N��W?̢��K��	�Q�Q>Y ��N>ma?��?�?�ʏ>���h?��=�����}Y��Q]��A��2=E=�W?^R@%Tl�=��l��e��@��ؾ0iG?�&E@�y��a�}�����K?%Tl���G>�9l� ��>�,;��}Y��>>��VԼ�+u?hݏ�_3_�Ţ.?��G>�ޗ>�$���@=���?�X?�z>��X�¬�>���?����z;��+��O4�J�����>���?lȾ�� ?p׋�:]��տ�7�?Y;y�ǂ��	�g�!���q��?��*�?�͛?,V@��,h���>��>ZW�lk��۲>�A�����>ԽG?���?�{� Ý>̢���$����;���>"�I���¿<�>��<�r�?Fwо�J�=�28�%�M��sX>0iG?ɓ?�?�� ?hݏ���t�ad@�W?1��?��[�,H��r�y+��?W�ɿ�ӯ��4	�P6�1�
����>�G*�:o�������;=&��>�\6>�қ���t�l��>lݿ�$��٣�	�Q�1Q�����Q�? "?�ȫ?L� ���R=`yD��7_�z� �:o��r�A�&��������*˳���-���1��3�?< ����@J��˝��d|�)�5?ma?��W�_3_��@xua�5@�G*�	�Q�D���	�.�B������}% �ʇ@����z;�������[��*�lk��y+��,��@��>У�:]�P6�0�?��{>�٣��,;�)N<
�U?B ½Z�����w��?�2�"�/��>���%�M��q~�Q>}>�?���>�˴??�'�:]�cZ����%?l��w4˾��+>E��bm?J�>=֗�u�_��ĝ�ad@��W�?K��v������aF�&��>I
���<MF��2�"�ؒ;?�!����	�Q��}Y�5�������\6>X�ӿ:]���?��G>�T��O4�	�Q����aF�X���4���ك�MF����>�����{>XD_>f�����>�@�Da?2=E=u�_���t�ڄ��*ߩ?���� ��>���Sux?aC+>K�ǿ*�?�ӯ�(){?2�"�3
۾d/�w4˾	�Q���������E���g�MF��(.����%?JJ�����S��M7�"����=ݰ�?��<�?�P6�o,����w4˾	�Q����>�d�>�ߧ��|��p׋�_3_��]�?I}'�v
�?��?�q~�]?�ii>-�?�4���û?MF��)*>r�>�9l�)4?��ؾ��@��e�4��>z[	��ͫ��ξ��Ln?W�?��?ԡ�>�w�?/�=��*�f��>�'������r�>���>��x�v������>ɓ?�M�>'0о(��D���¦]��S4?̢���٣�	�Q���@��t�L�n���x>u�_��4	�P6�1�
����>�G*�:o�������;=&��>�\6>�қ���t�)*>�3N�&D�=w4˾
��>���>	(@�z>��WA�?�P6�,h��]��>�����V���?���?�b�R�/��ͫ���@¦]��OQ?����K���$�]?B�h�>ma?x�?��-��Y�?�4�����>�O4�:o���q(����J$��s�ҋ?�R4?�C�3
۾���><t&@v���D���\�?'�?3��=�ҋ?l�p=��R=�w����@?�٣�����.?Q����%���w��ԽG?
��?ʏ>I}'�&��?��x���=�q(��{�wd|?��?��<l�p=��d>�|?�$�%�M��QB@E���t���?2=E=�ӯ�	�������ƀ>̢����=�h�g��ဲ����@��>u�_����>2�"���?b���%�M�	�Q����=���?#��>*�?m�G���-���p� Ý>�X��G*��q~��.?Ռ�?�Z�=��*���:]��C��3N��7_���=��ؾ����d|�پ��|���ԓ�l�p=cZ��/��>�+��G*����g��B�M?�n��@;�У��'���ۚ�o,�Y;y��S�?,H��g��IO�P��?���b�?�$�>��>��?�G*���+>E���>>�wd|?���>��L� ��߰�G��>Y;y��٣�,H��)N<	�.��ɿP��?��7?�?¦]�@&D�=���?h~¾lk��B�h�*g��˴?V�7��?B����J�=�$���-?�q~�)N<�w�?&��>Z���hݏ���t��x?벪�N�&? ��>j�n>���y+��@�7�? ��?��-�2�"�z;��T٣�,H��D���"��C�>�z�?���>uyE�TXi?�ǿ�u?z� �&�Sux?�{������\��ҋ?�>k��`yD�|��?w4˾J���M7��X?�z>�\6>?�'�D����x?l�R���?�O4����?�}�?E�?Т?�4��}% ����a�?�g?96R���>�h�r�"���Da?G�K�7ʏ>9Ӟ���1�1�
���@?��@h~¾��?A@ @j.�>Ӥ2?���>l�p=�i�?�w���h?w4˾J��r��t�?2����<D�����>����G>XD_>��=���>e�?j]??2=E=��:]����� Ý>̢��w4˾�۠��q(��Q]�/�=@��>��:]�V��?1�
����>)4?(��=0iG?�6?_O(?��*��_>l�p=^�N��W?�#���$���q~��M7�B�h��M��\6>m�G����>�x?K��W�?��>�h�g���U��n�z[	����b�?�ۚ���o��$���x���+>���>�{��	��*�?��Y1�?a�?�g?��q��$��J���.?�p;@'�?��X�?�'���t��?�w��W�?��?	�Q�ԡ�>*�t?��?G�K�7ʏ>�ξ�i�?��o�ۓ�?�$���G?˝�ŀ��&-��@;�m�G�ά?���>�U@�7_����>��E��5���B�����?�ӯ��R4?����6]�?�X���=���E���t�\�>���>(����K?����k��?��q�<�|?UX�>Չ�?�X?U����쮾�қ�D����K@�3
۾�>��>J��)N<y0F?�+?wm�?�ӯ�:]��C�֤$@< ���٣�D��ԡ�>�=�(>���?�'���-�Ţ.?���?�7_��٣��,;�r�"��wd|?z[	�(��9Ӟ��Z?l�@�>w4˾�,;����?��J@�+?�4��m�G�:]�I�?\������?�٣�f��E���t� ��?I
�7ʏ>D�����d>�W?�l.=��>xB>���=!�.?��>�쮾��_3_�������>�ޗ>w4˾v���0iG?�@/�='0оV�7���t���R=��>&D�=��=���>Չ�?M�@�`,?�4����w��ĝ������B�=�4E�%�M���ؾg��ဲ�lȾ�\6>hݏ��b�?�ۚ����?�����٣�bN�r������V�ݰ�?hݏ��ξ�xu@��#[?��x�:o�����	�.����?W�ɿ0Г?D������o���>z� ����ԡ�>1䃾�cy�E��ԓ����>�j(@�q~�W�?�!@ZW�E��ဲ��@�@;���<���?B���6]�?�7_���=�S�(�2@\�?�S?�7�?m�G��ξP6���?���>XD_>����?aC+>\�>���u�_����>�]�?	�οW�?��:o��r�*�M�]V?��\��!@O?�x?Ln?96R���[�f����@�	�.�����3��ԓ���-�k�����u?��x��i?�}Y�*�M����2=E=��l�p=cZ���q~��ޗ>�$���q~�g�� <~<v"˾Z����w?O?��p�,h��N�&?��&��}Y�*�M����X�7ʏ>�?)P=?�J�=��@?�$���S�A�_?aC+>\�>�@;�f��>c�c?��@�-?W�?r��?��+>����y6��\Y@<#�x�?�?�@ Ý>��?K��,H��E���Q]��+?������ξ�{��i @�����x�ZW�D���*�M�q�M��˴?(��MF���x?�q~��h?%�M�(��=ԡ�>}>�?l��?�4��*˳���t��Z?z;�W�?��=��c@E��*�M��Q�?����W?y>������%?< ��K��ZW��q(��e��@q��z�?ԽG?�>(.���OQ?Y;y��٣�(em����>��;=�cy�]��?�ك�c�c?��1�s�#[?%�M���ؾ���="��[�꿳�x>V�7�
��?�i�?xua��3?�G*�;j<r�J�l�?����g?�'��2�"�l�R���?XD_>��ؾ�.?�{�ڎ׿�\6>���>�n@�{���%?��G>�J0@X�½�}Y��==Do>��x>��>L� ��C�bԟ���?ǂ������}Y��ى����E���W�(){?vׄ��h�<b���w4˾}���?A����z�<�>�ҋ?L� ����?�3N�W�?��x�X�½E���t�� �?G�K�*˳�D����{�G��>�#��K���$���@�A�&�#��>��@¬�>l�p=2�"�`yD���{>�������R�?2��>`�Y>�+u?ԽG?MF��Fwоr�>JJ����x�,H���}Y�����7���X�?�'���-�V��?���?z� ��$����	�.�O%F?�@;��û?��t�Fwо�s����@?w4˾T'F�r�	�.��n�:�S?Ճ?ά?��p�벪�W�?��x�xB>���>�d�>`�Y>������=9Ӟ���ƿ�4��]��>��[�v���D����Q�?ɔ5�2=E=��D���TXi?z;��$��G*�c�>r��d|�'�?�\6>��W��ξ�����?�9l�XD_>f��Q>�=cq�?�FC?���>�?�a�?1�
���?��x�"�>��@���wd|?R�/���w�uyE��$�>��d>�>�W�>��=)N<�w�?���?<�>hݏ�:]�¦]��OQ?̢�����]Q�E���ى��,���7�?��W�:]�TXi?K���u?��x�6ң=����*��B���\��_>ά?�C���B?����w4˾f��r��y6��S?�R�?���?MF��B����4����?�$��	�Q����=��ؽ�⛿�����7?l��?����h�<b�����=��ԡ�>A���l��<�>��Ͼy>���l�R�|��?��=�۠�]?���?�z>2=E=7ʏ>uyE�I�?,h���?��?���>����y6�-�?��X�F��?9Ӟ����G��>��q�K��۠�0iG?}>�?�ߧ��|��m�G��?�)*>���>�W�>�q~��F�?m�+@&��>�|��m�G���t����=U�5�N�&?�!:@�]Q�˝�����"R���X�(�����?�$�>U�5���{>-�@����� <~<��?<�>��7?l�p=����`yD���@?w4˾���q(�aF�J�H��� ?��<�b�?��?��>�$�)4?��>ԡ�> <~<?�쮾���=uyE��Z?K����?%�M�}��q(�����8?�s�}% ����>�@�3N���?�W�>c�>���?���?��	@a�}�t�	@(){?��?z;���@?w4˾��;���>A0>O%F?a�}����>(){?�@��K�?w4˾,H��r��t�?!��ȫ?_3_���d>����>XD_>ZW��������7�3��=���=�R4?�Z?��B?��?��?X�½���>�d�>ꉒ?���>f��>(){?vׄ��OQ?����'}�?�*�g��*�M�|��d?(�����?>2�������������E����1S��@��>�ͫ�9Ӟ����W?Y;y�K���q~�lk���f���2=E=V�7�:]�����o,����K��S��M7�1䃾�Da?�FC?}% �	���%Tl� Ý>Y;y����sX>lk���?�����X�V�7�D���ڄ���ƀ>�l.=�٣�͓\�!����=>Z˿ "?�ӯ�ʇ@@��?�3N�K�? ��>������y+�/�=a�}�ԽG?l��?(.���h�<�9l�S"�?�G?�M7�"��/�=7ٛ?���=O?��1��C�?������,H�����> <~<NAc�E�У���t�cZ��	�ο�>�2�?&���@��>>��M�>-�����?��>2�"����?< �����?������ى��,�� "?X�ӿ��>k���3N�l���٣�,H���M7�A�&�������x>�����?vׄ��h�<b���w4˾}���?A����z�<�>�ҋ?L� ��?���&��?��x���?r�*�M�\�>�쮾��<Y1�?�x?������?��x��h�˝��t�cq�?�@;�ԽG?uyE�)P=?/��>1��?�G*��,;����?�k>?��?7ٛ?����-�)�$@�s��XM@��=��+>�q(���cq�?�@;���7?���>>�:@�0��p��?��=J��r�aF����?a�}����=��-�@��?�OQ?��?K���:�r��e�&��>�%m��ԓ��ξl��>U�5�1��?��,H���}Y�J�l�_�����*��_>l��?�K@���>����z� �T'F�˝�ŀ��lȾ "?�ԓ�L� ��]�?�o��|��?K��ZW�r�����D�>3��=�ȫ?��>cZ��=��W�?��=f�����?a>�>?�\6>��?0�̿2�"��h�<�28�w4˾��$�Q>�>>������*���>MF���@l�R��?w4˾ҧ<?�}Y�*�M�-�?�s��'?�4	��Z?�i @���%�M����>���?V�R@�w@ "?���=�R4?���� Ý>�28��٣�'I�>���=Q���V����s�m�G��R4?����B?����O4�����a����,��ma?�ԓ���t���>3
۾W�?��=f�����>�~]? ��?��x>����-���/��>������=�U5��}Y����KH=�<�>(�����>2�"�/��>���%�M��q~�Q>}>�?���>�˴??�'�:]��@xua�v
�?��=X�½A�_?Y'?H9�?�4��Ճ?�4	����/��>������[�}�g�Q]���#��d?(��E9�?��ƿz;��9l��G*���V����=���?�ο"�@��,V@�$�>��?QԾS"�?M��?BcW@�&E@wd|?�%m���?D����K@���?�T�N�G�&��}Y�A:�='D^�}j���ӯ��'���C�A�>�T$��	�Q�lk��2��>�B����V�7��?��G?�0���3?w4˾6ң=r�J�l��+?��X�7ʏ>��-�^�N�I}'���G>w4˾�,;����>A0>U�����??�'�:]�(.���;�?������x��������?aC+>�ȿ]��?��W��b�?�m�?I}'����?N�G���+>]?�*�4��>�O��'?MF��¦]�AH@b���w4˾bN�����ى��޾�˴?w뿽��>�������̢��z� �f��!���"��%K8�@��>V�7�l�p=�����D�>�F���٣��h�D����?��E���g�D���ʏ>U�5�N�&?��x��,;����>aC+>�+?:�S?w뿽D���Fwо=��Y;y���>f��lk����;=/�=��*���Ͼ�ξ%Tl�$��?�4E�w4˾��D���]+�?B���<�>��W�9Ӟ�I�E@C> ���@ ��>�R����=��ؽ�@0ҿ�g?�ξ�����W?|��?z� ��܍>�M7��Q�?�L��G�K�u�_�:]����=�w��N�&?���?�U5�˝�a���G�><�>��<�b�?Fwо�C�?�����٣���ؾ�F�?ဲ�
?��7�?}% �c�c?����6]�?�7_�)4?��m=��������u?@��>�ك�:]��x?��#[? ��>�������ŀ����:3��=u�_�E9�?l��>���?������=��=g����cq�?@��>Gˣ?��t��$�>�4��+�?w4˾��?Sux?Q��?��C>�쮾m�G���-�>2��U�5��-�����;g��A�&��(�Ӥ2?x�?9Ӟ���p�	�οN�&?�O4�ۺ/�g��A�&���2=E=�ȫ?0�̿������G>�4E�w4˾J���}Y���\�>���>��w���t���1��w���l.=w4˾J��Չ�?Y'?!&�=�d?p׋���t����?�q~�|��?���>��m=E��aF�wd|?�%m�m�G�l�p=3�?�OQ?v
�?�G*�f��r����� �?�%m�f��>:]�T6��G��>96R���ZW�Q>���p��wm�?��<��>��p��h�<�$�w4˾���lk��"G�?lȾZ���V�7���t���1���>̢���G*�J��0iG?Ռ�?l'�>wm�?m�G���-�ad@xua�2�@r��?ZW��}Y�1䃾U@�?-���ӯ�,V@�ۚ���&D�=�٣�v������?e�?��3��=u�_��?��@1�
�ۓ�?���>���>����y6�� �?�4���͛?�4	���p�xua��X����h��.?�=�n���X����=�'��wh@l�R�ۓ�?��=;j<����*��w@�%m���>�?��{���B?�����W�>?���>"��s�s��d?��L� ��a�?U�5�]��>K��S����=!����:'��}% ��R4?��ƿ��?�!��N�G��$�r����L�n�hs@u�_�9Ӟ�2�"���?�#���G*�������@��r� ��2=E=��g����?����z;������x�(��=!���y+�����+u?�ԓ��ξ��p��q~�JJ��K��۠���8@���?��7�Z�����:]�¦]���Y;y���x����Q>1䃾v�#��|��}% �uyE����=��Q��?��(��=���?Z��?g��=�|����?l�p=�{�o,������x���;��@��t�5u�ma?����t�@��?-ֿ]��>��x��*��M7�Q���s��z[	��ك��'��P6�l�R�#[?K��j�n>r��t�5u��|��w뿽c�c?�l3@��懡?��x�,H��D���ဲ��@�Ȱ�(��	����$�>(���h?��:o�����aF��N%� "?��'?ά?l��>�;�?������=xB>g��y+�cq�?��x>�ҋ?��t����0�?�M��W�@j�n>�F�?Ȱ?�?�@;���D���¦]��OQ?d/���w�+?!���*�M�ɔ5�Ӥ2?���=��K?(.��=���9l��G*��:��R�?9
?D�> "?��>�4	�2�"�l�R���G>w4˾ۺ/�lk��!��z房z[	���'?��K?�K@�/��>�>�*�?��ؾԡ�>��?Т?�|����M@_3_��K@�z;����w4˾h~¾˝��>>��n�'0о��D���%Tl���%?�X��٣��$�g��M�@%K8�@��>7ʏ>��>�C���>��q�w4˾�$����=�d�?�����X���W�MF���C� Ý>�>�$��v���lk��a>�>/�=�+u?��Ͼ_3_��K@�r�>�28���x�:o��lk��ဲ����@��>��Ͼ��-�¦]�G��>�M�����h�g���?��C>��?���'��k����o��>��v������=	�.��֬��s�7ʏ>c�c?�C����?������x��$�Q>����> "?�ӯ���t�cZ��`yD���?N�G���$����!���,���쮾x�?�?��ۚ�`yD�n N?w4˾,H���M7�	�.�i����d?�ȫ?�R4?)P=?Զ��懡?��x�v���!���A�&�U�_�@��>��'?
��?��?�0���$��٣����lk���Ѿ5u�-����<ʇ@ʏ>	�ο���?%�M�X�½A�_?�Q�>�S?�� ?V�7�D���2�"���G>��q�w4˾X�½!���.��?e��쮾��W�MF������1�
��K@K����A]?�X?�M�>=֗��y@��t��ۚ���G>�9l�z� �6�@�0iG?�᾽]����?�ӯ���K?I�?��>��?��=��!����=?��\�}% �O?��p��q~��u?h<U?	�Q�lk���m��j.�>��x>Gˣ?ʇ@@��?벪�W�?��=���E�����&��>
]ڿx�?l�p=�����J�=��q�w4˾;j<�.?��;=��8? "?x�?D���B����S4?�7_� ��>�@=���="�I�D/?E�?�'�:]�P6���%?�F���$���.>E���Ѿ�ߧ����>��O?��ƿz;��9l��G*���V����=���?�ο"�@��,V@� @�o��懡?��,H��r�����/�=�@;����>_3_�>2����G>������x��,;�r�a����y��ݰ�?V�7�(){?%Tl����$�z� ��h�Q>q
¾�y���� ?�ӯ��>�G?`yD�#[?�٣�UX�>�R�?y{@'�?c���s�a@:]�k��?;	@�>��=Rz?�}Y�2��>:	?!�\�Q@_3_���R=��o���{>��x��S����?�|?��7��|�����>O?B����s��W�?<�|?��m=g��	�.�L�n�Z���w뿽�j�?k��?U�5��3?��x�f���}Y��e��Da?}j�����=�'��)*>xua� �??@�$�˝�ŀ���B?Z�����O?Fwо�g?�X��G*���?ԡ�>���?��7���*�hݏ��'���߰�벪������A!侏�@Pu\@c�>3��=V�7�L� ����C�?�#���٣����Չ�?��ؽcͿ "?�����?�7@�w��懡?��?�.>���=�VJ>�@
]ڿ�w?:]��K@��"@���w4˾��D����X?P����x>�ӯ���t���ܿ�C�?QԾN�G�Er�?�}Y�R�>���j��?hݏ�O?�$�>� �?�T�%�M��h�lk����;=���>�@;��ԓ���-�l��>��o���?��bN����>a>�>֛r���\�Ճ?O?��1���_?�����G*������=y+���?�d?m�G�:]���d>�4���u?��[����g��A�&�����@��>��'?9Ӟ�ڄ��r�>d/����h�Sux?ဲ��	��2=E=Ճ?�b�?I�?I}'�懡?�W�>�$���?a>�>���?�쮾^Ƴ?O?�Y�?�S4?]��>�2�?
��>g���?�F@2=E=w뿽�4	�TXi?����Q��?w4˾�۠�˝�ŀ��ܷ=��4��¬�>c�c?cZ����+@�9l��G*������}Y��m��q=2=E=?�'�	���P6���?�ޗ>��>;j<)N<�m���z>�����y>%Tl��J�=�28�w4˾�q~�lk��Q���j.�>�R�?w뿽�?��C�U�5��>z� ��$���?�|?g��= "?}% �:]��K@�Ln?Y;y����sX>]?� �?����\6>V�7�D������ Ý>�M��w4˾�۠��R�?A����s��Ȱ�*˳��'��Fwо�4���>%�M�J��Sux?ဲ�ڎ׿-����Ͼl��?T�P@I}'�K�?<�|?�@=r� <~<�@p����w?���>�]�?Զ��K�?XD_>f��E���>>����?�쮾x�?�R4?������G>QԾ�����>D��� <~<��7�G�K���?:]�vׄ�����d/��W�>���D����f��'�3��=��Ͼʇ@����z;��9l���x�?����{���C>��\�u�_�:]�^�N�Y��?JJ���ϛ?v���)N<�?>�c?�˴?���>���Z?3
۾��?%�M��S��R�?ɓ?�T�>�4�����?�V��?�o����?K��	�Q�r�	�.�&��>��\���@��>Fwо��@�����٣��,�?E��a���0��7ٛ?��˿��>Fwо��o���G>z� ��,;�Sux?�?�jd>�+u?u�_�:]��@3
۾W�?��[�:o��˝��E��g��==֗����4	�(.�����?�!����?,H���}�?۲>��k��R�?�ك���@��ƿ��B?������x���˝��ى��ȿ�=%@��ʇ@��1�1�
��+���J�ｲ}�?A0>�gX�]��?w뿽�4	��]�?�D�>��?��=�q~�g��	�.�� �?a�}��_>	����ۚ���G>�9l�z� �6�@�0iG?�᾽]����?�ӯ���K?�������l.=�٣����.?�ii>�+?Ӥ2?x�?D���)P=?������?��[�ۺ/��q(�	�.����=-��Ճ?L� �2�"����?< �����?������ى��,�� "?X�ӿ��>l��>���?�F����x��,;�D�����;=&��>��\���g�uyE��Z?z;��>�G*��.>���
9���?�s�ӯ���-�ʏ>Զ���3?<t&@�۠��}Y����em���|��u�_��j�?�?xua���?��?,H�����=�?q�?-��ԽG?9Ӟ���1���?b���#Xo�h~¾E��5��� )���%m�p׋��?�(.���ƀ>�4E��G*��7?A�_?Q����R���?f��>��K?)*>B��?96R��G*�f��D���"���u?ma?p׋�MF��2�"��W?����K���sX>�M7�1䃾�.�;���>��	���Fwо�-?�!��K�����M7��Q�>g��=���>�қ�L� �>2�� Ý>�X�%�M�F s�)N<�6?6Eڿ�+u?}% �l��?Ţ.?��W�?�!:@���E���e�?��*�¬�>O?)P=?��_?�$���x���E����?c����ӯ�_3_�¦]���o���{>��!����ŀ��Ӌ�@��>?�'�ʇ@���G��>�M��z� ��q~���?2��>�s��Ȱ���˿�'���C��OQ?̢����x����Չ�?Ȱ?"R�2=E=��>(�?�x?�q~���?w4˾D��r��ى�x�z[	���W����?%Tl�eU�?���w4˾�܍>ԡ�>�Q�>��;��?x�?Y1�?��p�z;��ޗ>w4˾f��lk���p;@/�=�@;�?�'�:]�ʏ>�4��N�&?XD_>���>�}Y�J�l�s������W?���?P6�3
۾�X���x��.>)N<!���,����*��ك�_3_�Fwоa�?����K��܍>r�5���/YO�@��>��ϾO?^�N�Y��?JJ���ϛ?v���)N<�?>�c?�˴?���>���P6��ǿn N?z� �!�]?A:�=�	��<�>�ӯ�ά?)P=?��G>�X���-?'I�>�@Ȱ?-�?�|��}% �9Ӟ��������?Y;y��$��(��=����*��M�>�s�f��>�?�Fwо=�����S"�?���>���?E�?�R?�4�����ĝ�TXi?z;�1��?K�
Ao�$@��D@%��?��@@a�}���?�/��@��?벪� �?�$��:o��r��e�P��-�������?2�"�o,������[�v���E��*�M�j.�>2=E=�w?uyE�)*>T��?����XD_>��+>���=��ؽ��H?3��=�ӯ��4	��@l�R�W�?��>j�n>D����{�U@�?a�}�hݏ�L� �P6��ƀ>�X���?�۠����=M�@�.�;�쮾���'��%Tl���X?����$���@=���=%��?B�����x>V�7�9Ӟ�2�"��ƀ>b����O4��q~��q(��f��z>@��>7ʏ>uyE�1��=��������F s�Q>}>�?�?/�B�@¬�>S�:@>��������O4��:����>�)�?�����?u�_�E9�?��1�1�
���@?��@,H�����?O@j.�>Ӥ2?f��>l�p=����W?l��w4˾X�½���=��?`���쮾���?�k���W?����$����m=g���m���u?�FC?��'?��-������|?b���K��,;�ԡ�>ဲ�*�x�ma?�_>��>k�� Ý>����w4˾����}Y��ى�پ��d?�ﾞ�K?��>���?�$�z� �	�Q����aF�Ds�=E����=�ξ�$�>����$��O�>�.>���?�)�?O%F?�Ȱ�m�G�D����@��G>W�?�$�����E���f��X�?}j���ȫ?��>�x?��G>�$���x��,;����>۲>��?��\�(��9Ӟ�����z;��>K��:o����@��>>�������(���4	�����G��>&D�=�O4��i?�q(�!���M�>�쮾�Ǵ@�ĝ���1�/��>�#���G*��,;�0iG?�m��p�X�ma?hݏ���>Ţ.?eU�?��q�K��,;�!������N>'0оu�_��ξ�x?1�
��ޗ>��>��+>���?� �?�Da?-����>�ξ�K@��J�=�9l�w4˾�S����J�l�Y'���FC?��Ͼ(){?�����?�����O4�(��=]?"��(�R�<�>���ξ��1��OQ?�����G*�X�½Q>�f�@�>:�S?V�7���t���d>�|?�$�%�M��QB@E���t���?2=E=�ӯ�	���2�"�ؒ;?�!����	�Q��}Y�5�������\6>X�ӿ:]��C�o,���{>�٣�h~¾BcW@�6?N����\6>Ճ?�n@a�?z;��>h<U?J��E���d|� ��?E���W�y>�G?U�5�W�?%�M��R���@�9@'�?-��?�'���t�ʏ>/��>&D�=��x�j�n>E���t�?@��>�ك��ĝ��a�?�q~���?��=h~¾E��J�l��z>�4��}% �O?���l�R��#���O4�͓\�Sux?2��>��G�!@��?^R@��1�1�
���@?��@,H�����?O@j.�>Ӥ2?f��>l�p=���?=����?�g�=�R��q(�q
¾�t�>�s��7?�j�?����k��?������x�f�����"��B ½'0о��w��?��]�?l�R��u?�٣����>r�������V�'��}% ����?k��xua�n N?��=J��Q>%��?9>3��=��>��-�%Tl����$�z� ��h�Q>q
¾�y���� ?�ӯ��>(.��l�R�1+���x�~Q��lk��A�&�p�X�:�S?}% ��?)�$@��?ۓ�?�!@X�½�}Y��{�8'9@W�ɿ¬�>�ξ%Tl�*ߩ?̢����	�Q�r�5���lȾwm�?����徖�1���1+�XD_>�q~�lk���m���S��@��>�ك�y>��1�
������=�۠���?qE ?�������>��l�p=�K@� Ý>�4E�K��6ң=�R�?G�@5u�:�S?p׋�l�p=)�$@�s��XM@��=��+>�q(���cq�?�@;���7?���>vׄ���?����K�򾖍���q(��Q]��z>�+u?}% �D���B����?�M��N�G��$��}�?M�@�z>a�}���9@�ĝ�cZ��z;����> ��>�h����>�]@&��>z[	�u�_���t��$�>U�5��?��=T'F�!����f���k�2=E=u��?��.@2�"�;9�>̢��K��xB>���"����k��+u?m�G����?�����OQ?v
�?��Y;&?���?��7@/�=�%m�u�_���t���`yD��4E�z� ��sX>���>�f��EӾa�}��ك���t��Z?��B?��?��?X�½���>�d�>ꉒ?���>f��>(){?��1�����l.=XD_>��ؾr�J�l���/=�s�m�G�L� �^�N��ƀ>����XD_>�S����>�*��jS� "?��W����>k���3N�l���٣�,H���M7�A�&�������x>�����?^�N�I}'��ޗ>��x��S�]?Q����cy� "?��?�?�$�>���h?z� �!��}Y�5����ؿ'0о?�'���?��?�W?JJ����>�sX>��?�??��X���uyE�>��w���������]Q��}�?qE ?1;.�B�@�W?^R@2�"��7�?����w4˾v�?����ى�=����� ?��W��ξ�]�?1�
�1��?�G*���ؾ!����U�lȾ�Ȱ��ȫ?c�c?�� Ý>�#����;j<D����Q]�Ǜ�2=E=(��l�p=l��>��_?96R�XD_>��=g��A����.�?���?�'�:]�^�N�a�?�!���G*��:��}Y�J�l���`��d?��(){?¦]���?Y;y�K����ؾ���?1䃾j_��Ӥ2?Ճ?:]�@��?K����?�ϛ?(��=g��	�.����>E���@���?�Z?�w����?��[�ZW�˝�)c���A��G�K�?�'�c�c?��>3
۾W�?��=f�����>�~]? ��?��x>����-���/��>b���w4˾�,;����>�*�l��=�� ?u�_�9Ӟ���R=� �?96R�K���m=)N<qE ?l��G�K�Ճ?uyE��C��OQ?��q�����@)N<A0>g��=a�}�E�A@��t���ƿ;9�>< ���$���۠����>A�&�O�P��??�'��n@��1�1�
���@?��@h~¾��?A@ @j.�>Ӥ2?���>l�p=^�N�3
۾96R��$��d�x�r�5����ؿ�|��}% �c�c?¦]���%?������>�G?lk���Ѿ%K8�:�S?����K?��p�U�5�&D�=��	<�?1Q��A���j.�>�|�����?:]��{�G��>�#��K���$���@�A�&�#��>��@¬�>l�p=����߹�@������J��˝�5�����7���?p׋���>�{� Ý>������x��܍>�AQ@qE ?(0�ݰ�?��>c�c?�ۚ�1�
��F����	�Q����>�Ǔ>��]��z�?�ӯ�:]��C�K���M���O4���V�0iG?9
?c�ۿ<�>��Y1�?���?/��>W�?��>�R�����*�cq�?�4����l�p=�C�Զ����?�٣��h�E���ى�B���3��=m�G�L� ��{�˹�?̢����>������=q
¾�(�`�?�û?��?TXi?U�5���?�M@�sX>r��U�wd|?Z���Ճ?Y1�?I�?=��|��?��=�h���@�*�M��q?���m�G�O?�x?I}'��h?L65@,H��r��d|��x>��*��ك����?cZ��a�?�9l�w4˾!�E������N�����X������?�!@/��> �?�G*�}�r�A�&���?�����MF���Y�?3
۾�X�K��J������Q]��?-��}% ��'��ڄ�������28��G*��q~���?"�I��x�Ӥ2?��?O?TXi?�q~���?N�G���$�Չ�? <~<��6<�쮾Ճ?�>����o,����K��S��M7�1䃾�Da?�FC?}% �	������J?�����$����m=��@��U��M�>Z���¬�>�'���w�/��>�+��G*���$����=�)�?��|s_@�ӯ��n@¦]�Ln?����-�@,yM?���>۲>��?�+u?���>_3_��ۚ���o��$���x���+>���>�{��	��*�?��Y1�?����l�R��l.=�٣�ZW�!���Y �/�=z[	�¬�>��-�@��?�OQ?��?K���:�r��e�&��>�%m��ԓ��ξڄ��`yD�Y;y�z� �ZW�˝�*�M�ɔ5��\6>����t�B����"3@̢����-?f��r��ى��?�7�?�ك�O?��1���_?������x��۠�D����>>�g��=�7�?��w��'��Ţ.?���$���>(��=�M7�A����+?'��x�?�ξP6�,h��]��>�����V���?���?�b�R�/��ͫ���@l��>z;��>K�򾖍����@��e��?2=E=f��>l�p=�?��W�?XD_>;j<g�ᾬz>-����>Y1�?1�뿖����T٣��6~�)N<���?���+u?�ȫ?�n@��ƿ�J�=���z� ���V�E����� )���e)@��c�c?2�"��W?����K���sX>�M7�1䃾�.�;���>��	�����p��s����@]B�@�]Q�Ǟ@/��@n��p�����g���?¦]�G��>���N�G��*�E������x�I�Ӥ2?�ӯ�:]������>w4˾&�Q>q
¾]�P���X�u��?��K?�K@��g?����K��!侒�@qE ?�u?��?�w?MF��k��r�>QԾ��J��lk��q
¾��̽]��?��>l�p=�C�q@E@�9l�z� ���?���� �?�,���|����	������=����G>�O4��$�lk���{�'�?���w뿽	������=���$�w4˾bN����>A:�=?�*��e)@x�?^R@�߰��-?�4E�w4˾:o�����?A����,�*�?f��>_3_�cZ��6]�?������x��@=!���y+�'�?���>�қ�uyE����o,��-�K���R�D���ဲ�/�=<�>u�_�:]�2�"�Զ���ޗ>z� �D��˝����lȾz[	�.�1@�ξ�j(@z;�v
�?XD_>����}Y�	�.��C�?�����>��>Fwоr�>JJ����x�,H���}Y�����7���X�?�'���-�2�"��W?����K���sX>�M7�1䃾�.�;���>��	����$�>G��>W�?w4˾�@�.?��?�z>'�� �@�'���{��W?����'}�?��=BcW@��p?�+��@;�У�:]�)P=?l�R���G>%�M�xB>)N<�=O%F?G�K�7ʏ>�ξ%Tl��a:@Y;y��G*�!侱��>���u��z�?��O?FwоI}'�W�?w4˾;j<)N<��ؽ�+?�7�?w뿽�ĝ��i�?�w���h?w4˾J��r��t�?2����<D���P6� Ý>�l.=��?ZW����>S@j.�>-��V�7���t�l��>U�5�+�?G�?�$�r�*�M�������X���l��?��p���B?������[�xB>r�����@��FC?��:]������h�<�#���O4��h�˝��U���k�a�}��қ��?�2�"��u?�28�w4˾�q~�D���*�M��q-�E�V�7�9Ӟ�B����o����@?w4˾�$�g�f��M�>��*��g?��t��C�1�
��$�%�M�	�Q��}�?�X?�u���?��Ͼ�?¦]���_?JJ���٣�h~¾�q(��y6��q-�<�>�ӯ�L� �@��?��o�&��? ��>�BB?����*��+?z[	���<l��?vׄ�U�5��7_�w4˾�q~��F�? <~<H�"�@��>p׋���t�P6�o,��!��z� �h~¾���ဲ����\6>���ξ2�"���B?�!���G*�	�Q�)N<�y�?q�>�˴?����-���p�/��>b����O4���;lk���d�>��7��|���қ���t�����k��?����)4?xB>Sux?�w�>~�?��?���>�?���p��?�#��K���q~��}Y���U����� ?V�7�uyE������Z@�9l�N�G�lG�?�}Y��U�긹�7ٛ?p׋��?�k����?�7_��$���$�)N<�Ѿ�t�>`�?�_>���>¦]���B?����O4�h~¾)N<�w�>/
�E��ӯ��'��ڄ����_?�#����h?c�>���?�ii>e�*�?��<��K?l��>��W�?��=�.>��@�)�?O%F?}j����Ͼ�ĝ�ʏ>r�>�����x��q1?�M7��Ǔ>�,����*��ӯ���>vׄ���1+��$���i?�q(��m��!��>hs@}% ��?�w?�@��o� �?�)u@xB>]?�k>?&��>I
�0Г?j��@�K@�� �?�4E����i?lk��9
?e��-����Ͼ:]��K@�6]�?Y;y�w4˾��=����U�5u�ma?�ԓ��ξ�����D�>�F���٣��h�D����?��E���g�D������=K��1��?K��]�@lk��1䃾�gX�p���.�1@��t��?K����@�ϛ?�i?r�	�.�\�>0ҿ��<��.@���?K����@?��x�����M7��U�n �>'0о���4	���d>�J�=��q�%�M��sX>�}Y�"��&��>Z����ӯ��'������6]�?�X���=���E���t�\�>���>(����K?�$�>lݿ|��?���h���?2��>/�=�� ?m�G���t����C�?�#���٣����Չ�?��ؽcͿ "?�����?�� Ý>�7_���'I�>���?���?P��}j���ك��?�ʏ>��B?��q�z� �v�������y6���C>�4��x�?D��������h�<���z� ��q~����="���7�3��=(���ξ)*>\���N�&?��x��,;�˝��E��ˍ[�Z������=ά?�@U�5� �?8,����ؾ�}Y��f�?0ҿ��>��K?^�N������X�w4˾���!����>>�����ma?0Г?�R4?�$�>�3�?�28�H�7?;j<����d|�cq�?:�S?��W�(){?��p��s����@]B�@�]Q�Ǟ@��@n��p�����g���?�Z?�s����?z� �&�˝�a����y��c�����(){?�$�>�����?��?��>˝�����e���*���Ͼ�j�?cZ��Զ����{>XD_>!����5����8���|����'?ʇ@%Tl��J�=�28�w4˾�q~�lk��Q���j.�>�R�?w뿽�?����?,h��|��?N�G���ؾ�}Y��t�j.�>-��w뿽9Ӟ���1��4���>��x�6�@���?�]@l��2��-��@:]�P6�=���+�w4˾�۠�r��Q]��z>E�hݏ�D���k���3N�QԾ�W�>�h�)N<�m���������>��g�l�p=2�"�o,������[�v���E��*�M�j.�>2=E=�w?uyE�(.��`yD��ޗ>)4?	�g�D����VJ>�� �@��>u�_�E9�?TXi?��>�����-?'I�>���?�k>?�L�?���f��>�?���1����M��K��h����?�ii>5u����>w뿽�4	�k����%?������܍>��@��y6��L���|����uyE�>2���C�?�7_���=6�@�!����f�Q}ӿ*�?��w�ά?)P=?I}'�]��>�G*�6ң=����y6�:	?�s�(��l�p=��1���>�����٣�6ң=�q(��d|�|�Ӥ2??�'�(){?������?���N�G��q~���@��t���`��쮾m�G�uyE���p�,h��N�&?��x���V���@�X?lȾ3��=��?��K?�x?"�?n N?�G*����lk��"�𾈺���4����O?��ܿ��96R���x�͓\�)N<� �?��&���?^Ƴ?S�:@�x?��G>�$���x��,;����>۲>��?��\�(��9Ӟ���K?�h�<���?w4˾�E���U�x�|��Gˣ?
��?�@�ƀ>1��?��=J��r����@'��ԽG?y>Fwо��?��{>��x�J���}Y��ii>����@��>��W���t��������?�F����x����g���*���@��>?�'��ĝ��K@���G>�28��O4��h�A�_?�{�s�s��\6>��ϾD�����R=��?�4E��������@�r�:gf�<�>�û?���?�m�?=����?)4?���>��?e�?�w@�@;�/��?D������?�?ۓ�?L65@;j<r�	�.�?�w����@�)@cZ��	�ο�>�2�?&���@��>>��M�>-�����?��>�C�r�>�!���O4�������=�{�5u����>p׋�	����{�Y��?�!���G*�������?��;=]�P��z�?(����?��1�
������=�۠���?qE ?�������>��l�p=)P=?/��>1��?�G*��,;����?�k>?��?7ٛ?����-���ƿ��?�����O4��@=)N<�d�>;a��`�?��W�MF��cZ��`yD�W�?�G*�q��>0�>@	(@�u?<�>��@�?�2�"���o��u?�٣�v���0iG?A���ܷ=�ma?�͛?O?�G?����+�?w4˾v����}�?�?g��=��X��W?y>��p�o,���q�K��,;��5 @g�5@��7�z[	�V�7�:]��?����懡?��Rz?���=A�����>��X���7?(){?l��>*ߩ?��@z� �-��@Q>9
?�����+��Ն@�ĝ�B����o����@?w4˾�$�g�f��M�>��*��g?��t�cZ���o���>%�M��۠�)N< <~<@�ؾma?ԽG?��K?�C������T٣�h~¾D�����@�>�z�?¬�>:]��m�?����N�&?��[�6ң=r���8R#>W�ɿ}% ��N��l��>��|��?��!�Sux?�r���2=E=��@�>��I}'�2�@���x7A��@�
9�/�=0ҿ��q@��t�B���_�?1+�%�M���=!����ii>D�>@��>V�7�9Ӟ���?�?�?�T�x��?���Z��?g��=,8�"Ԍ@_3_��{��ƀ>̢���٣�?r�J�l��׽<�>�_>:]���R=�S4?���?�٣��3
?D���E�?j.�>�@;���W�D����߰�K�������O4��6~�˝��ى�V����7�?�͛?Y1�?k��r�>QԾ��J��lk��q
¾��̽]��?��>l�p=��>1�
��>%�M��$�Sux?�y�? ��?<�>u�_�:]��K@��ƀ>̢��w4˾������=y+���@��>hݏ�_3_��� Ý>�#����;j<D����Q]�Ǜ�2=E=(��l�p=����3N��M����=7p�@��@��d|�/�=��*�u��?l�p=����K���l.=��>�܍>Q>9o�?����-�����=��t��@l�R�W�?��>j�n>D����{�U@�?a�}�hݏ�L� ��a�?o,���?��=ZW���?�Ǔ>*g��4���W?�j�?�m�?I}'����?N�G���+>]?�*�4��>�O��'?MF���G?���?l��XD_>	�Q�0iG?�Ǔ>�Da?������=�>�@U�5��?w4˾���r�	�.�cq�?�⿗ӯ��'����K?�������?%�M��.>���?�ҍ?�Da?��X�Ճ?uyE����=ؒ;?W�?�J0@�BB?�R�?��J@�w@���E�A@�'��TXi?��>�����-?'I�>���?�k>?�L�?���f��>�?��C������{>�٣�&�Չ�?�ii>���>�|��}% ��ξ�K@���?�7_���[�,H���}�?qE ?lȾ��X�hݏ�D������?�񱿡�?z� ��:����=�ᾮZ�=�4�����=�>���OQ?�X��٣�;j<���=�Q�?�,��2=E=p׋�:]���>K��W�?�$��	�Q�Sux?��@'�?z[	���D���ʏ>G��>1+��$��!�����{�g��=G�K���Ͼ:]�%Tl����̢����x��$���?z��?�GK��Ȱ�m�G���-��������?懡?w4˾�BB?�@��N@g��=��*��_>�'������NY�?&D�='}�?X�½���>9
?&-�a�}�m�G���t���d>Ln?��G>�W�>���.?��?���?�7�?x�?�b�?B����J�=�l.=K��h�g��"�I�'�?�� ?hݏ��ξ^�N���o���བ$��j�n>E��J�l���7� "?�ӯ�9Ӟ�ʏ>I}'�&��?��x���=�q(��{�wd|?��?��<l�p=(.��U�5��X��O4�ZW�E��5���B����\6>(��D���Ţ.?��>#[?�G*�,H�����>Y'?��?�+u?��:]�Ţ.?�ƀ>l���G*�J���.?/b=@��?Z���?�'���t�vׄ�o,�96R����۠�!���1䃾�(��\6>��W��'��ڄ����������[��U5��}�?�Ѿ����d?p׋���?�x? Ý>�F��K���$��q(����M�>=֗�?�'�:]�k�����?QԾ�@�.>���>e�?�$T>a�}���W�uyE��@��W�?�*�?h~¾Sux?�=��8?
]ڿ��uyE�)P=?�i @�l.=�$�����F�?�5C@�u?�s�m�G���t���ƿ�4��O%@" Av����.?Ȱ?1׆<�� ?p׋�l�p=��d>o,��9l�%�M�6ң=ԡ�>�=&��>�\6>Gˣ?�>�ۚ� Ý>�����O4�������@��Q]�����P��?0Г?��t�����1�
���{>XD_>�$���?۲>�M�>���w뿽L� �ڄ���"3@����w4˾X�½���=	�.����ݰ�??�'��R4?(.����G>d/���	�Q����=�?�� �3��=�ӯ�l�p=����-@d/��٣�'I�>g�ᾬ���z[	�p׋��ξ��	W@n N?��x��i?]?S@\�>'0о¬�>D���)*>Զ��N�&?��=X�½)N<Y'?��?�|��p׋���t�B����S4?�7_� ��>�@=���="�I�D/?E�?�'�:]��e�?�o����@���<�R����A�&�\�>��\�hݏ����?TXi?�ƀ>���w4˾���>D���� �?�n���u��?9Ӟ�cZ���q~��ޗ>�$���q~�g�� <~<v"˾Z����w?O?%Tl����$�z� ��h�Q>q
¾�y���� ?�ӯ��>���e@�M��w4˾�.b�lk��2��>�(�@��>�ͫ���>�x?�3N��M�����@,H��!���A:�=�k!@a�}��_>D����@U�5�懡?w4˾(��=lk���ii>� �?�����<_3_��a�?�OQ?�F����>UX�>���=q
¾U@�?��X���@MF���G?���&D�=�٣��J^?r�J�l�/�=-�����>�R4?��p�&)��@?�O4��*�Q>Y �긹�E��_>�R4?^�N��3N�QԾz� ��,;�!���y+��,�<�>w뿽�?�cZ�������$�w4˾�,;�)N<�y�?�f�>�\6>w뿽:]��ۚ��S4?b����O4��]Q�r��Q]���ܿ*�?V�7�O��?���������@z� ����@�M7���B ½����!@�ĝ�������G>QԾ�٣��S�lk��"G�?�,��3��=V�7�:]���R=xua���?���2�@�q(�y+��u?0ҿ�!@�ĝ�� @l�R�v
�?�٣�,H��E���U�wd|?�w��u�_����>�m�?����N�&?��[�6ң=r���8R#>W�ɿ}% ��N������g?���%�M���$�E��a���NAc�<�>?�'���>�$�>(���h?��:o�����aF��N%� "?��'?ά?�K@�z;�1+���	�Q�����y6�q�M��˴?hݏ��4	������>�F����J��W&@2��>/m��Ȱ��Ȼ�D����{����?̢���k�@�BB?�Ο@G�@cq�?��?m�G�MF��>2����%?̢���$����;Sux?�r��3ֿ��?У�O?��� Ý>�!���ϛ?�$��}�?�r��޾�O��ԓ�D�����?�J�=��G>�G*��3
?g��1䃾�2>�s�(����>Fwо`yD���G>�G*�J���.?�k>?O%F?�z�?}% ���-�ʏ>��>�T� ��>�@=D���
�U?G�>c���V�7���t�Ţ.?xua�懡?w4˾J��r�y+��+?��\���g�:]�k��?����#[?�O4�}�)N<�m����n?-��ԽG?�ξvׄ�z9@�����G*�v�?r�5���ѩ&�P��?��g����>�$�>�����?��?��>˝�����e���*���Ͼ�j�?cZ��"�?JJ��z� �v���E�����/�=�@;���W�:]��ۚ��|?d/��2�?�q~�!����f�;�����?�ӯ�ά?)*>=����?����ؾr�5���!V���x>�_>Y1�?>2������9l�z� �	�Q�Q>Y ��]���d?V�7��R4?Fwо�W?�!��w4˾�$��}�?��ؽ��:�S?w뿽��>B��� Ý>�28� ��>xB>�M7�qE ?�Da?�%m�u�_��ĝ�%Tl����Y;y��٣����Q>q
¾�N%�3��=w뿽��-���d>=���h?�$��������?�d�>P��3��=��7?���?ʏ>/��>&D�=��x�j�n>E���t�?@��>�ك��ĝ�k����?�7_��$���$�)N<�Ѿ�t�>`�?�_>���>�G?��o���?N�G��$�@a>�>������?¬�>��>�?�"3@n N? ��>�q~�Sux?}>�?�Da?}j����9Ӟ�Fwо��o���G>z� ��,;�Sux?�?�jd>�+u?u�_�:]���K?�4����?�O4�&�!����Ѿ�S?Z�����<�ξ(.����o���?��x���G@���>ဲ�Hh⿦��>��?
��?�G?G��>��q���-?��m=A�_?\�?-�?�s�u�_���-��K@��J�=�����٣��,;��@a>�>5u��� ?0Г?c�c?P6�T��?l��w4˾�$�g��qE ?����E�p׋��?�¦]�Ln?����-�@,yM?���>۲>��?�+u?���>_3_���>��?���K���f�?�}�?E�?�Da?a�}�7ʏ>��t��$�>������G>w4˾;j<D����U�k���%m�?�'��4	��x?�o�����?K��X�½�5 @
�U?�#��I
������?(.����%?JJ�����S��M7�"����=ݰ�?��<�?��K@��ƀ>����w4˾��@D���
9���6<����ӯ���t�B���/��>�28���=�R�r�	�.��Da?�����ϾD���k���q~��ޗ>�$����ؾ�q(�y+��A��@��>�ӯ�ά?%Tl����$�z� ��h�Q>q
¾�y���� ?�ӯ��>)*>���?�$���H@Y;&?Չ�?;�)@�@2=E=��%@�'��B���G��>�>�٣�f��g��A:�=�z>ma?hݏ�_3_���?�h�<�$���=	�Q����?5�3@��?��*���:]���p��S4?�#��%�M�	�Q����=1䃾�n��\6>x�?Y1�?����/��>��q�w4˾�$�Չ�?�"$@;�����*���W�MF��ʏ>�g?��q��O4�;j<)N<�y6�!&�=2=E=�ԓ���t��m�?��#[?�O4�r4�?��?A0>,�[?<�>Gˣ?MF����ܿ��96R���x�͓\�)N<� �?��&���?^Ƴ?��8@�ۚ����+��O4��۠��}Y��ى�.Ũ��R�?V�7�c�c?�K@�/��>�4E���x�X�½!����������z�?u�_�_3_�^�N�a�?�!���G*��:��}Y�J�l���`��d?��(){?ʏ>3
۾n N?��[��q~�D���aF�ѩ&���*���?_3_�k���ǿ&D�=��x��:����>�=�C+�ma?��'?��>TXi?�J�=�h?�G*�,H��)N<"�I���?�� ?u�_���t�2�"��3N��$���x�h~¾E��Q���lȾ3��=p׋���t�k��?3
۾�ޗ>%�M�q��>ԡ�>1䃾�nC?�s�f��>���>�$�>Y��?�>��[��]Q��������H�"�'��p׋��ĝ�l��>�ǿ�>w4˾,H��(�2@�X?/� ?:�S?Gˣ?_3_����l�R�+�?�G*�"�>E��5���Pܺ�@��>?�'�c�c?��?1�
�懡?�G*�X�½�M7��y6�&��>��x>hݏ���K?Fwо6]�?����z� ��q~���?
9��ߧ����u�_�y>Fwо�����$��$��6ң=��@�1䃾�+?��?��<uyE��Z?`yD�n N?/y*���ؾE���ى��Ԅ�G�K��қ�c�c?cZ�� ��n N?%�M��q~�Չ�?A0>�J$��˴?�w?�4	�2�"��w����{>%�M���m=W&@!-@����2=E=�W?�?�K@���%?̢���٣�!��q(��y6�\�>��?���=��t�I�?�J�=�@�S�?,yM?r��t��Da?p����ҋ?���?2�"�I}'��>�٣�X�½���?y0F?\�>:�S?��W���t�k��a�?���z� ���lk����;=�i�'��p׋��ξ��Ln?< ����>:o��lk���f��M�>�쮾hݏ��ĝ�¦]�0�?������ZW��R�?aC+>&��>j��?��>MF��¦]���_?�9l�S"�?��ؾlk��!���GK� "?��W��?k����?< ��K����ؾ0iG?�{��L��Ӥ2?��(){?k����n@����͑�?�sX>E���d|�����"�@V�7�y>��1���1+�XD_>�q~�lk���m���S��@��>�ك�y>l��>o,��ޗ>#Xo�h~¾�q(�Y �'�?�s�ӯ��'��%Tl���%?�X��٣��$�g��M�@%K8�@��>7ʏ>��>¦]�T��?�9l���v������A�&�ܷ=�<�>У�uyE���R=��G>���w4˾J������>>�&��>-���ͫ�uyE�Fwоo,��4E��W�>!�D����VJ>��C>@��>���ĝ�2�"��u?�28�w4˾�q~�D���*�M��q-�E�V�7�9Ӟ���p��J�=�>%�M��q~���@��e�����@;����>]=@cZ���w��]��>K��*�W&@
�U?3L��z[	��w?���?��d>Y��?������[�����0iG?�Ѿɔ5�Z���?�'���-���>���X�XD_>(��=���?�d�?�S?�Ȱ�(��D�����d>Ln?��G>�W�>���.?��?���?�7�?x�?�b�?P6��4���h?�٣�N4@E��J�l��3ֿ�4���w?l��?�$�>	�ο��?�O4�v���˝�������>'����Ͼ:]���?��@�$���=��D���q��?���?�7�?V�7��?�k���|?����K���q~�g��q
¾'�?*�?(����t�a�?	�ο�ޗ>�G*�D��Ƅ]@B�M?:	?��?u��?_3_��߰�&)�JJ��z� �(em�\,|@m�+@���2=E=�ͫ�:]�2�"��ƀ>< ���G*��S�)N<Q���~�<�>�ӯ�L� �i:ο�3N��!����F s����?\�?�ZI�7ٛ?Ճ?^R@��?K��1��?�*�?UX�>E������+?��X��ҋ?(){?�����B�=�T�K���$�!���ဲ��z>�|����Ͼ	���cZ���ƀ>��q���x�bN����?�?��㾛|��*˳�9Ӟ�P6�=����{>%�M��$�Sux?�@�`,?-����%@��-���1���_?���K��:o��)N<A0>q=<�>��W�:]�P6��s��K�?�O4��S�)N<!��e�辛|��ԽG?���>�{��W?����'}�?��=BcW@��p?�+��@;�У�:]��C��OQ?̢����x����Չ�?Ȱ?"R�2=E=��>�?�{�K��d/��٣��q~�]?�VJ>k��2=E=u�_��'��>2����>�#���٣�ۺ/�E�����z房�z�?(��c�c?�x?=��1��?��-?UX�>�}Y�aF��u?�쮾��?���?���h�<b���z� �v����q(�B�h�������x>p׋��?����5@�7_��٣��3
?g�>>��n��쮾��w�:]�FwоI}'��$����$�����ى��,����X���Ͼ:]�%Tl����l.=�$��2ӿ��?.��?|�N��쮾X�ӿY1�?^�N� Ý>�#��K���$���@�ii>ŜE=ma?¬�>��-��7@1�
���?K��h~¾lk��"�I�U@�?0ҿ�ك�	�����1�$��?�#����x����g�Q]�n��� ?��˿ʇ@3�?��Q��?K��:o�����=2��>�+?a�}����?���p��a:@< ����x�f��!���!�����-��?�'�MF��%Tl����l.=�$��2ӿ��?.��?|�N��쮾X�ӿY1�?%Tl�$��?�#���$��	�Q����=���(�wm�?hݏ�ʇ@�]�?r�>�X���>6ң=Q>��;=cq�?-��w뿽L� �a�?��?�l.=��?��Q>y0F?���?���m�G���-��G?�3N����>z� ��S�˝��d|��x>Z���ԽG?��>^�N��J�=�>��x�;j<�q(�"G�?���>'0о\�Q@l�p=���?3
۾K�?��x��۠��}Y��y6�n�?��*��_>��>a�?�0����?�$��v���r��Q]��,���@;��=@���?���a�?����ǂ���聿)N<�r�O�$��@��Ͼl��?�$�>��>1��?w4˾*�@�}�?��?��7����ľ@:]��@�񱿜��?�G*�J��D���!����?�w��"Ԍ@�?�>俩3N���q��G*���m=�.?
�U?�$C��=%@u�_�E9�?�l3@��懡?��x�,H��D���ဲ��@�Ȱ�(��	����K@�-ֿN�&?�O4�}����>�X?]�P�7ٛ?���?�K@����?< ��w4˾�S�!����ii>������ ?���?���`yD��4E�z� ��sX>���>�f��EӾa�}��ك���t�2�"��-?����K���sX>lk��B�h��.�;���>��D����a�?1�
���?��x�"�>��@���wd|?R�/���w�uyE���R=/��>�7_���x��R��M7�z��?��7��4����<�?����� Ý>�M���$��:o��lk��A:�=lJ��`�?���=l��?����z;�������[��*�lk��y+��,��@��>У�:]�ʏ>K��W�?XD_>X�½Q>z��?O%F?�|���ӯ�:]�cZ��߹�@����z� ��S�˝�5�����7��˴?�ك���>�$�>�q~�N�&?h<U?
��>D���	�.�-`)�z[	�7ʏ>���?��d>6]�?�?z� ����@D���ɓ?�	�������M@D���P6��C�?�F���G*�:o��!�����?��=�s�V�7�:]��G?1�
�#[?K���R��q(��>>�9>��X�?�'�:]���>����G>��>f��0iG?y{@�Da?���hݏ���t�2�"�C> ����>w4˾�S�!���
9��z>�7�?w뿽D���)P=?z;����>XD_>;� ?Sux?a>�>�{�>��\���>��>>��-?����8,���$�lk�����⛿�XB@hݏ�l��?�C��S4?����ǂ������lk��aF��	���� ?�ԓ��ξ�]�?K���@	nF@v�����@�Ce?�Q�?�%m��͛?���?�G?��o���?N�G��$�@a>�>������?¬�>��>B�����|��?�٣���=r�5���3L���쮾u�_�
��?�]�?�񱿜��?��>���>E���U�?'0о��:]���d>Զ����?)4?}�����U�՜���|��¬�>���?k��/��>��q��٣�X�½���=��?`��E�(���?�ad@`yD�ۓ�?�W�>��=D���Y �-�?��\������>^�N��W?�ޗ>�M@��N@��@�@gd?-��(��:]�V��?=��ۓ�?k��>;� ?D���!��\�>�s��?Y1�?���/��>�!������+>A�_?���?��k���x>V�7�9Ӟ��?`yD���?��w�+?r��d|�O%F?�|����?�?���?�h�<|��?%�M�}�r�A�&����?'����Ͼ�'��k��I}'��$���ZW�Q> <~<5u����>u�_��4	������%?�����$���h�Q>!��l��2=E=�ӯ��ξ)*>��?��@���0�@���>�Ǔ>NAc��O��E@�'�����?�o��#[?h<U?,H��r��U�wd|?'��Ճ?�'��¦]�˹�?����ǂ����{@)N<!�����2=E=��g��?�3�?U�5�1��?w4˾q��>���>A:�=��?�Ȱ����4	���?�0��Q��?��[��.b�˝�ŀ�����a�}���g����?�{����?̢���k�@�BB?�Ο@G�@cq�?��?m�G�MF�����?��o�5@�٣���m=r�aF�� �?�%m�u�_�D���V��?	�ο��@?���h�r��>>�&��>��*�Q�@���>�x?����$���>'I�>!������ �?��X�(���ξ�$�>l�R�N�&?��x���ؾD�����r�~�Z�����?�r�?¦]���>b����_+@bN�g�Q]��,�� "?��Ͼ���>��1����M��K��h����?�ii>5u����>w뿽�4	�¦]������#���٣�v����q(�A�&�B���3��=(��:]�B����o��]��>��[��q~���@��y6��M�>�쮾7ʏ>uyE����G��>��q�K��۠�0iG?}>�?�ߧ��|��m�G��?�l��>/��>����K�򾟄$�E������]V?��\����4	�@��?K����@?w4˾,H���}Y�*�M�"R�c����W?�?k��/��>��q�0Sk@h~¾��?��@O%F?���>�ك���t��$�>1�
��h?K��T'F�r�J�l��y����x>7ʏ>l��?�C�`yD���G>�$��v����q(��r��,�� "?V�7���t���1�`yD��4E��٣������M7�	�.�
��|��m�G��)@���J�=��q�K��@=�.?�5C@q�M��쮾��Ͼ��-�%Tl���?̢��K�������?1䃾z房�|��(���"�>�@`yD�5@XD_>��+>���=ဲ�cq�?�Ȱ�hݏ����>Fwо���?̢��%�M�,H������d|�%K8�*�?f��>Y1�?�K@� Ý>�l.=���?�R�D���!�.?��]�2=E=7ʏ>O?k�����?QԾ�@�.>���>e�?�$T>a�}���W�aXM���-ֿ]��>��	�g��.?2��>��C> "?f��>�j�?���?3
۾�@��?ZW�r��>>�wd|?�%m���<�R4?TXi?�q~��3?�G*�X�½g���w�>O%F?��*���w�:]�l��>��]��>�O4�!�r������N%�z[	����?TXi?z;�1��?K�
Ao�$@��D@%��?��@@a�}���?�/��l��>��B?d/���?�sX>�q(��d�>���?��X�V�7�uyE�B�����G>�F���$��J������y�?�S?���>u�_�:]�l��>벪��ޗ>���}�!����f�!V��쮾0Г?Y1�?���?�4��W�?��>�.>)N< <~<wd|?'�� �@�>TXi?�0��W�?\�?J���q(��e�gÀ�G�K���W�l��?)P=?��G>��G>XD_>�@=lk���r��u?�w�����?�ξ(.���4���9l���[�ZW�lk����#���Ӥ2?7ʏ>�ξ2�"��w��JJ��%�M�:o��@Y'?e��>V�7�9Ӟ���>���?�$�z� �	�Q����aF�Ds�=E����=�ξ������B?�K@��x�j�n>A�_?A@ @�z>}j��m�G�	�������˹�?�#����=v�����D@E�?�v��<�>u�_�    �?��o�+�?�G*���r�*�M���6<c������>    �K@�1�
�̢��K���$�lk����_�v���x>�w?    a�?Զ����?XD_>�,;��q(�Y �&��>'��V�7�    V��?��>W�?�٣�h~¾�M7� <~<��?�|��(��    B����v�?�#����x�h~¾E���d|���2=E=��    �$�>#=�>�3?�k�@���F�?���?���?Z�����    2�"��w���$�w4˾,H���F�?aC+>X��:�S?��    ���l�R�+�?�G*�"�>E��5���Pܺ�@��>?�'�    �,@���� �?��=��=lk�� <~<-�?
]ڿhݏ�    2�"�G��>Y;y���?J��Sux?N��?�n2�'0о��g�    ���0����@��A?Z��@E�������z>'0о��)@    FwоU�5�&D�=K��	�Q���?ɓ?��C>�d?��Ͼ    �C��7�?�!���٣�(��=����U������� ?�ԓ�    2�"��3N��T�>��;!���q
¾��E�p׋�    �]�?K����@?N�G�T��?�R�?��;=tkV?<�>Gˣ?    *
dtype0
}
train_datasetTensorDataset%train_dataset/TensorToOutput/Constant*
output_shapes
:	�	*
Toutput_types
2
�(
$train_target/TensorToOutput/ConstantConst*�(
value�(B�(�	"�'  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  @@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  @@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  @@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  @@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  @@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  @@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  @@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  @@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@   A  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  @@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@  �@                                                                *
dtype0
w
train_targetTensorDataset$train_target/TensorToOutput/Constant*
Toutput_types
2*
output_shapes	
:�	
�
train_dataset/Zip
ZipDatasettrain_datasettrain_target*
output_types
2*%
output_shapes
:	�	:�	*
N
�
(Input_Input/Zip/Initializer/MakeIteratorMakeIteratortrain_dataset/ZipEstimator/Train/Model/Iterator*1
_class'
%#loc:@Estimator/Train/Model/Iterator "a
Saver/Constant:03Saver/WithControlDependencies_1/Identity/Identity:0Saver/Group_1 (5 @F8}�Y�K       �	��	��|��A�A:@</Users/ashwinravishankar/Work/tmp_foo/udx/tmp/log/model.ckptƮ�