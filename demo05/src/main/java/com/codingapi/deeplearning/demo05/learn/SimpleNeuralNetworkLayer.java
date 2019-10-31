package com.codingapi.deeplearning.demo05.learn;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.List;


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
 */
@Slf4j
public class SimpleNeuralNetworkLayer {

    private INDArray w;
    private INDArray b;
    private INDArray a;

    private INDArray dw;
    private INDArray db;
    private INDArray delta;

    private INDArray x;

    private boolean isOutLayer = false;

    private int index;

    private List<SimpleNeuralNetworkLayer> layers;


    public SimpleNeuralNetworkLayer(List<SimpleNeuralNetworkLayer> layers,
                                    int in, int out, int index) {
        this.index = index;
        this.layers = layers;

        w = Nd4j.rand(in,out);
        b = Nd4j.rand(1,out);

        log.info("index:{},w:shape->{}," +
                "b:shape->{}",index,w.shape(),b.shape());
    }

    public SimpleNeuralNetworkLayer(List<SimpleNeuralNetworkLayer> layers,
                                    int in, int out,int index,boolean isOutLayer) {
       this(layers,in, out,index);
       this.isOutLayer = isOutLayer;
    }


    public INDArray forward(INDArray data){
        log.debug("forward before=> {}, w.shape->{},b.shape->{}",index,w.shape(),b.shape());
        if(index==0){
            x = data;
        }
        int length = data.rows();
        INDArray z = data.mmul(w).add(b.broadcast(length, b.columns()));
        a = Transforms.sigmoid(z);
        log.debug("forward res => {}, w.shape->{},b.shape->{},a.shape->{}",
                index, w.shape(), b.shape(), a.shape());
        return a;
    }


    public INDArray back(INDArray data,double lambda){
        log.debug("back=> {}, w.shape->{},b.shape->{}",index,w.shape(),b.shape());
        if(isOutLayer){
            delta = data;
        }else{
            delta = data.mmul(layers.get(index+1).w.transpose()).mul(a.mul(a.rsub(1)));
        }
        INDArray _a = index==0?x:layers.get(index-1).a;
        dw = _a.transpose().mmul(delta).add(w.mul(lambda));
        db = Nd4j.sum(delta, 0);
        log.debug("back res=> {}, delta.shape->{},dw.shape->{},db.shape->{}",
                index,delta.shape(),dw.shape(),db.shape());
        return delta;
    }

    public void updateParam(double alpha) {
        w = w.sub(dw.mul(alpha));
        b = b.sub(db.mul(alpha));
        log.debug("updateParam=> {}, w.shape->{},b.shape->{}",index,w.shape(),b.shape());
    }
}
