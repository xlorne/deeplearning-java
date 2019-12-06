package com.codingapi.deeplearning.demo07.learn;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


//关于运算Nd4j下的运算符号介绍:

//INDArray mmul INDArray 是矩阵之间的相乘 (2,3)*(3,2) => (2,2),两个矩阵需要满足矩阵相乘的形状 (A,B) * (B,C) => (A,C)
//INDArray mul INDArray 是矩阵中每个值相乘 (2,3)*(2,3) => (2,3),两个矩阵必须是相同的shape
//Transforms.sigmoid sigmoid函数
//INDArray broadcast 是将矩阵的数据做扩充，例如(1,3)的数据通过 broadcast(4,3)就变成一个(4,3)的矩阵，数据都与第一行相同
//INDArray add 是矩阵的加分运算，必须是（A,B）+ (A,B)形状的。add方法也可以加 一个数字，就是对所有的项都加该数组
//INDArray transpose 是矩阵的转置运算
//INDArray rsub 是对矩阵的逆减法，例如 [[2,3]].rsub(1) = >[[1-2,1-3]]
//INDArray sub 是矩阵的减法运算，方式同加分运算
//Nd4j.sum 是 对矩阵的row,columns是求和，设置为0是对所有的columns求和，1是对columns求和，不设置是全部求和


/**
 * @author lorne
 * @date 2019-10-31
 * @description 简单的神经网络层
 * <p>
 * 涉及到的参数:
 * lambda 正则化参数
 * alpha  学习率\步长
 * <p>
 * w的初始化叫做W的权重
 * z = w.Tx+b
 * <p>
 * sigmoid是激活函数
 * a = sigmoid(z)
 * <p>
 * L是损失函数
 * L = −y⋅lna−(1−y)⋅ln(1−a)
 * <p>
 * 用于计算各层dw、db导数的delta
 * delta(l) = delta(l-1)* w(l-1).T*(a(l)*(1-a(l))))
 * <p>
 * 各层的w导数
 * dw(l) = a(l-1).T*delta(l) + lambda*w(l)
 * 各层的b导数
 * db(l) = delta(l).*ones()
 * <p>
 * 梯度下降公式
 * dw = dw-alpha*dw
 * db = db-alpha*db
 */
@Slf4j
public class DenseLayer implements NeuralNetworkLayer {

    //W权重，通过Random初始化
    private INDArray w;
    //b，通过Random初始化,相当于之前说的theta0
    private INDArray b;
    //a,正向传播结果
    private INDArray a;
    //w的代价函数导数
    private INDArray dw;
    //b的代价函数导数
    private INDArray db;
    //改层的输入值
    private INDArray input;
    //是否是输出层
    private boolean isOutLayer;
    //当前层数索引
    private int index;
    //所有的网络层
    private NeuralNetworkLayerBuilder layerBuilder;
    //激活函数
    private Activation activation;

    /**
     * 该网络层的大小(in,out)
     * out 标示输出值的大小，可以等同于神经单元的数量
     * in  接受到的数据大小
     */
    private int in, out;


    @Override
    public void build(NeuralNetworkLayerBuilder layerBuilder, int index) {
        this.index = index;
        this.layerBuilder = layerBuilder;
    }

    /**
     * 初始化权重参数
     */
    @Override
    public void init() {
        //w 实际的维度就是 输入,输出值的大小
        w = Nd4j.rand(in, out).mul(Math.sqrt(1d/in));
        //b 是一个Vector 长度为out
        b = Nd4j.rand(1, out);

        //打印隐藏参数大小
        log.info("index:{},size:{}x{}", index, in, out);
    }

    private DenseLayer(int in, int out, Activation activation, boolean isOutLayer) {
        this.in = in;
        this.out = out;
        this.activation = activation;
        this.isOutLayer = isOutLayer;
    }

    private static DenseLayerBuilder builder;

    public static DenseLayerBuilder builder(){
        if(builder==null){
            builder = new DenseLayerBuilder();
        }
        return builder;
    }


    public static class DenseLayerBuilder {
        private int in;
        private int out;
        private Activation activation = new SigmoidActivation();
        private boolean isOutLayer = false;

        private DenseLayerBuilder() {
        }

        public DenseLayerBuilder input(int in, int out) {
            this.in = in;
            this.out = out;
            return this;
        }

        public DenseLayerBuilder isOutLayer(boolean isOutLayer) {
            this.isOutLayer = isOutLayer;
            return this;
        }

        public DenseLayerBuilder activation(Activation activation) {
            this.activation = activation;
            return this;
        }

        public DenseLayer build() {
            return new DenseLayer(in, out, activation, isOutLayer);
        }

    }

    /**
     * 正向传播
     *
     * @param data a,当index==0时，传入的实际是x,即a0 = x
     * @return 正向传播结果
     */
    @Override
    public INDArray forward(INDArray data) {
        input = data;
        log.debug("forward before=> {}, w.shape->{},b.shape->{}", index, w.shape(), b.shape());
        a = activation.forward(data, w, b);
        log.debug("forward res => {}, w.shape->{},b.shape->{},a.shape->{}",
                index, w.shape(), b.shape(), a.shape());
        return a;
    }


    /**
     * 反向传播
     *
     * @param delta delta 当是输出层的时候其实是 y-a(l)
     * @return 该层的delta，以及更新了dw,db值
     */
    @Override
    public INDArray backprop(INDArray delta, double lambda) {
        INDArray newDelta;
        log.debug("backprop=> {}, w.shape->{},b.shape->{}", index, w.shape(), b.shape());
        if (isOutLayer) {
            newDelta = delta;
        } else {
            //delta(l) = delta(l+1) * w(l+1).T*(a(l)*(1-a(l))))
            newDelta = (delta.mmul(layerBuilder.get(index + 1).w().transpose())).muli(activation.derivative(a));
        }
        //dw(l) = a(l-1).T*delta(l) + lambda*w(l)
        INDArray a = (index == 0) ? input : layerBuilder.get(index - 1).a();
        dw = a.transpose().mmul(newDelta).add(w.mul(lambda));
        //db(l) = delta(l).*ones() => sum(delta(l),0)
        db = Nd4j.sum(newDelta, 0);
        log.debug("backprop res=> {}, delta.shape->{},dw.shape->{},db.shape->{}",
                index, newDelta.shape(), dw.shape(), db.shape());
        return newDelta;
    }


    /**
     * 一次梯度后的更新参数方法
     *
     * @param alpha 学习率
     */
    @Override
    public void updateParam(double alpha) {
        w = w.sub(dw.mul(alpha));
        b = b.sub(db.mul(alpha));
        log.debug("updateParam=> {}, w.shape->{},b.shape->{}", index, w.shape(), b.shape());
    }

    @Override
    public INDArray w() {
        return w;
    }

    @Override
    public INDArray a() {
        return a;
    }

    @Override
    public boolean isOutLayer() {
        return isOutLayer;
    }
}
