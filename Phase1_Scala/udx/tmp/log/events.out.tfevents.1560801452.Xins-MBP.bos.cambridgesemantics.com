       �K"	��+�A�Abrain.Event:2%����`     ����	T�+�A�AJ��
*
1.12.0-rc0��
s
global_epochVarHandleOp*
shared_nameglobal_epoch*
_class
 *
dtype0	*
	container *
shape: 
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
Estimator/Train/Model/IteratorIterator*
shared_name *5
output_shapes$
":���������:���������*
	container *
output_types
2
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
#Estimator/Train/Model/Input/FlattenReshape*Estimator/Train/Model/Input_Input/Zip/NextEstimator/Train/Model/Shape*
T0*
Tshape0
�
\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/WeightsVarHandleOp*m
shared_name^\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
_class
 *
dtype0*
	container *
shape
:@
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
�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/RandomNormalInitializerRandomStandardNormal�Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/Initializer/Shape*
T0*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0*
seed2 *

seed 
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
YInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/BiasVarHandleOp*j
shared_name[YInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
_class
 *
dtype0*
	container *
shape:@
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
transpose_b( *
T0*
transpose_a( 
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
dtype0*
	container *
shape
:@@*Q
shared_nameB@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
_class
 
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/ShapeConst*
valueB"@   @   *S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/ConstantConst*
dtype0*
valueB
 "    *S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights
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
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/MultiplyMul�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/RandomNormalInitializer�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights/Initializer/Constant_1*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
T0
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
=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/BiasVarHandleOp*
_class
 *
dtype0*
	container *
shape:@*N
shared_name?=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias
�
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/ShapeConst*
dtype0*
valueB"@   *P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias
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
�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/MultiplyMul�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/RandomNormalInitializer�Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias/Initializer/Constant_1*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
T0
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
%Estimator/Train/Model/Linear_1/MatMulMatMul#Estimator/Train/Model/ReLU/Subtract$Estimator/Train/Model/ReadVariable_2*
transpose_b( *
T0*
transpose_a( 
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
%Estimator/Train/Model/ReLU_1/ConstantConst*
dtype0*
valueB
 "���=
�
%Estimator/Train/Model/ReLU_1/MultiplyMul%Estimator/Train/Model/ReLU_1/Constant.Estimator/Train/Model/ReLU_1/ReLU/NegativePart*
T0
�
%Estimator/Train/Model/ReLU_1/SubtractSub.Estimator/Train/Model/ReLU_1/ReLU/PositivePart%Estimator/Train/Model/ReLU_1/Multiply*
T0
�
(Input/Flatten/OutputLayer/Linear/WeightsVarHandleOp*
_class
 *
dtype0*
	container *
shape
:@*9
shared_name*(Input/Flatten/OutputLayer/Linear/Weights
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

seed *
T0*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0*
seed2 
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
%Input/Flatten/OutputLayer/Linear/BiasVarHandleOp*6
shared_name'%Input/Flatten/OutputLayer/Linear/Bias*
_class
 *
dtype0*
	container *
shape:
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
{Input/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/RandomNormalInitializerRandomStandardNormaliInput/Flatten/OutputLayer/Linear/Bias/Initializer/Input/Flatten/OutputLayer/Linear/Bias/Initializer/Shape*
T0*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0*
seed2 *

seed 
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
LEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/Sum_1SumNEstimator/Train/Model/Gradients/Estimator/Train/Model/Loss/L2Gradient/MultiplycEstimator/Train/Model/Gradients/Estimator/Train/Model/SubtractGradient/BroadcastGradientArguments:1*
T0*

Tidx0*
	keep_dims( 
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
VEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul_1MatMul%Estimator/Train/Model/ReLU_1/Subtract~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity*
transpose_a(*
transpose_b( *
T0*
_class
 
�
YEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/GroupNoOpU^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMulW^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul_1
�
}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/WithControlDependencies/Identity/IdentityIdentityTEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMulZ^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/Group*
T0*g
_class]
[Yloc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul
�
Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/WithControlDependencies_1/Identity/IdentityIdentityVEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul_1Z^Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/Tuple/Group*i
_class_
][loc:@Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/MatMulGradient/MatMul_1*
T0
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
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/SumSumVEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/MultiplyhEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU_1/MultiplyGradient/BroadcastGradientArguments*
T0*

Tidx0*
	keep_dims( 
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
TEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/MatMulGradient/MatMulMatMul~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_1/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity$Estimator/Train/Model/ReadVariable_2*
_class
 *
transpose_a( *
transpose_b(*
T0
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
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/ShapeConst*
dtype0*
valueB 
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
SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/ReshapeReshapeOEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/SumQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Shape*
_class
 *
Tshape0*
T0
�
VEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Multiply_1Mul#Estimator/Train/Model/ReLU/Constant}Estimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/SubtractGradient/Tuple/WithControlDependencies_1/Identity/Identity*
T0
�
QEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Sum_1SumVEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Multiply_1hEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/BroadcastGradientArguments:1*

Tidx0*
	keep_dims( *
T0
�
UEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Reshape_1ReshapeQEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Sum_1SEstimator/Train/Model/Gradients/Estimator/Train/Model/ReLU/MultiplyGradient/Shape_1*
_class
 *
Tshape0*
T0
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
REstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMulMatMul|Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity"Estimator/Train/Model/ReadVariable*
_class
 *
transpose_a( *
transpose_b(*
T0
�
TEstimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/MatMul_1MatMul#Estimator/Train/Model/Input/Flatten|Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/WithControlDependencies/Identity/Identity*
T0*
_class
 *
transpose_a(*
transpose_b( 
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
�
�Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias/GradientDescent/ApplyDenseResourceApplyGradientDescentYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias2Estimator/Train/Model/GradientDescent/LearningRate~Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/Identity*
T0*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
use_locking( 
�
�Estimator/Train/Model/GradientDescent/Update/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights/GradientDescent/ApplyDenseResourceApplyGradientDescent\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights2Estimator/Train/Model/GradientDescent/LearningRate}Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear/MatMulGradient/Tuple/WithControlDependencies_1/Identity/Identity*
use_locking( *
T0*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights
�
mEstimator/Train/Model/GradientDescent/Update/Input/Flatten/OutputLayer/Linear/Bias/GradientDescent/ApplyDenseResourceApplyGradientDescent%Input/Flatten/OutputLayer/Linear/Bias2Estimator/Train/Model/GradientDescent/LearningRate�Estimator/Train/Model/Gradients/Estimator/Train/Model/Linear_2/AddBiasGradient/Tuple/WithControlDependencies_1/Identity/Identity*
T0*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
use_locking( 
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
$Estimator/Infer/Model/ReadVariable_1ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
#Estimator/Infer/Model/Linear/MatMulMatMul#Estimator/Infer/Model/Input/Flatten"Estimator/Infer/Model/ReadVariable*
transpose_a( *
transpose_b( *
T0
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
$Estimator/Infer/Model/ReadVariable_2ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights
�
$Estimator/Infer/Model/ReadVariable_3ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias
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
%Estimator/Infer/Model/ReLU_1/ConstantConst*
dtype0*
valueB
 "���=
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
&Estimator/Evaluate/Model/Linear/MatMulMatMul&Estimator/Evaluate/Model/Input/Flatten%Estimator/Evaluate/Model/ReadVariable*
transpose_b( *
T0*
transpose_a( 
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
(Estimator/Evaluate/Model/ReLU_1/ConstantConst*
dtype0*
valueB
 "���=
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
dtype0	*
	container *
shape: *
shared_name	eval_step*
_class
 
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
^
Saver/ReadVariableReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
`
Saver/ReadVariable_1ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
�
Saver/ReadVariable_2ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias
�
Saver/ReadVariable_3ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/ReadVariable_4ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/ReadVariable_5ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/ReadVariable_6ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
Saver/ReadVariable_7ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
b
Saver/ReadVariable_8ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
b
Saver/ReadVariable_9ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
�
Saver/ReadVariable_10ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
Saver/ReadVariable_11ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
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
�
Saver/ReadVariable_14ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/ReadVariable_15ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
h
Saver/Constant_1Const*@
value7B5 B/_temp_f922d27f-a9cf-40ce-85e0-19f8806abd9c/part*
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
Saver/Constant_3Const"/device:CPU:0*
dtype0*
valueB
 "   
{
Saver/ShardedFilenameShardedFilenameSaver/StringJoinSaver/Constant_2Saver/Constant_3"/device:CPU:0*
_class
 
a
Saver/ReadVariable_16ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
c
Saver/ReadVariable_17ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
�
Saver/ReadVariable_18ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
Saver/ReadVariable_19ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/ReadVariable_20ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0
�
Saver/ReadVariable_21ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
Saver/ReadVariable_22ReadVariableOp(Input/Flatten/OutputLayer/Linear/Weights*;
_class1
/-loc:@Input/Flatten/OutputLayer/Linear/Weights*
dtype0
�
Saver/ReadVariable_23ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/Constant_4Const"/device:CPU:0*
dtype0*�
value�B�BYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/BiasBglobal_epochB@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/WeightsB%Input/Flatten/OutputLayer/Linear/BiasB=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/BiasBglobal_stepB\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/WeightsB(Input/Flatten/OutputLayer/Linear/Weights
Z
Saver/Constant_5Const"/device:CPU:0*#
valueBB B B B B B B B *
dtype0
�

Saver/SaveSaveV2Saver/ShardedFilenameSaver/Constant_4Saver/Constant_5Saver/ReadVariable_18Saver/ReadVariable_17Saver/ReadVariable_19Saver/ReadVariable_21Saver/ReadVariable_20Saver/ReadVariable_16Saver/ReadVariable_23Saver/ReadVariable_22"/device:CPU:0*
dtypes

2		
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
a
Saver/ReadVariable_24ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
a
Saver/ReadVariable_25ReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	
a
Saver/ReadVariable_26ReadVariableOpglobal_step*
dtype0	*
_class
loc:@global_step
W
Saver/Constant_6Const"/device:CPU:0* 
valueBBglobal_step*
dtype0
L
Saver/Constant_7Const"/device:CPU:0*
valueB
B *
dtype0
n
Saver/Restore	RestoreV2Saver/ConstantSaver/Constant_6Saver/Constant_7"/device:CPU:0*
dtypes
2	
J
Saver/Identity/IdentityIdentitySaver/Restore"/device:CPU:0*
T0	
j
Saver/AssignVariableAssignVariableOpglobal_stepSaver/Identity/Identity"/device:CPU:0*
dtype0	
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
Saver/ReadVariable_29ReadVariableOp=Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias*
dtype0*P
_classF
DBloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Bias
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
�
Saver/ReadVariable_30ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/ReadVariable_31ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights
�
Saver/ReadVariable_32ReadVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*o
_classe
caloc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
dtype0
�
Saver/Constant_10Const"/device:CPU:0*q
valuehBfB\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Weights*
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
Saver/AssignVariable_2AssignVariableOp\Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/WeightsSaver/Identity_2/Identity"/device:CPU:0*
dtype0
�
Saver/ReadVariable_33ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias
�
Saver/ReadVariable_34ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
Saver/ReadVariable_35ReadVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*l
_classb
`^loc:@Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
dtype0
�
Saver/Constant_12Const"/device:CPU:0*n
valueeBcBYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/Bias*
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
Saver/AssignVariable_3AssignVariableOpYInput/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Input/Flatten/Layer_0/Linear/BiasSaver/Identity_3/Identity"/device:CPU:0*
dtype0
c
Saver/ReadVariable_36ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
c
Saver/ReadVariable_37ReadVariableOpglobal_epoch*
_class
loc:@global_epoch*
dtype0	
c
Saver/ReadVariable_38ReadVariableOpglobal_epoch*
dtype0	*
_class
loc:@global_epoch
Y
Saver/Constant_14Const"/device:CPU:0*!
valueBBglobal_epoch*
dtype0
M
Saver/Constant_15Const"/device:CPU:0*
dtype0*
valueB
B 
r
Saver/Restore_4	RestoreV2Saver/ConstantSaver/Constant_14Saver/Constant_15"/device:CPU:0*
dtypes
2	
N
Saver/Identity_4/IdentityIdentitySaver/Restore_4"/device:CPU:0*
T0	
o
Saver/AssignVariable_4AssignVariableOpglobal_epochSaver/Identity_4/Identity"/device:CPU:0*
dtype0	
�
Saver/ReadVariable_39ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*
dtype0*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias
�
Saver/ReadVariable_40ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
�
Saver/ReadVariable_41ReadVariableOp%Input/Flatten/OutputLayer/Linear/Bias*8
_class.
,*loc:@Input/Flatten/OutputLayer/Linear/Bias*
dtype0
r
Saver/Constant_16Const"/device:CPU:0*:
value1B/B%Input/Flatten/OutputLayer/Linear/Bias*
dtype0
M
Saver/Constant_17Const"/device:CPU:0*
valueB
B *
dtype0
r
Saver/Restore_5	RestoreV2Saver/ConstantSaver/Constant_16Saver/Constant_17"/device:CPU:0*
dtypes
2
N
Saver/Identity_5/IdentityIdentitySaver/Restore_5"/device:CPU:0*
T0
�
Saver/AssignVariable_5AssignVariableOp%Input/Flatten/OutputLayer/Linear/BiasSaver/Identity_5/Identity"/device:CPU:0*
dtype0
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
�
Saver/ReadVariable_45ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/ReadVariable_46ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/ReadVariable_47ReadVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*S
_classI
GEloc:@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights*
dtype0
�
Saver/Constant_20Const"/device:CPU:0*
dtype0*U
valueLBJB@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/Weights
M
Saver/Constant_21Const"/device:CPU:0*
valueB
B *
dtype0
r
Saver/Restore_7	RestoreV2Saver/ConstantSaver/Constant_20Saver/Constant_21"/device:CPU:0*
dtypes
2
N
Saver/Identity_7/IdentityIdentitySaver/Restore_7"/device:CPU:0*
T0
�
Saver/AssignVariable_7AssignVariableOp@Input/Flatten/Input/Flatten/Input/Flatten/Layer_1/Linear/WeightsSaver/Identity_7/Identity"/device:CPU:0*
dtype0
�
Saver/GroupNoOp^Saver/AssignVariable^Saver/AssignVariable_1^Saver/AssignVariable_2^Saver/AssignVariable_3^Saver/AssignVariable_4^Saver/AssignVariable_5^Saver/AssignVariable_6^Saver/AssignVariable_7"/device:CPU:0
2
Saver/Group_1NoOp^Saver/Group"/device:CPU:0
X
ReadVariableReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	 "oOzw