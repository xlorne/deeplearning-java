package com.codingapi.deeplearning.demo05.learn;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;


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
 *
 * @author lorne
 * @date 2019-10-31
 * @description 神经网络反向传递算法实现
 */
@Slf4j
public class BackPropagationFunction {

    private INDArray w1;
    private INDArray b1;
    private INDArray w2;
    private INDArray b2;
    private INDArray dw1,dw2,db1,db2;


    private INDArray delta1;
    private INDArray delta2;

    private double lambda;
    private double alpha;

    private int batch;


    public BackPropagationFunction(double lambda, double alpha, int batch,int inputs) {
        this.lambda = lambda;
        this.alpha = alpha;
        this.batch = batch;

        w1 = Nd4j.rand(2,inputs);
        b1 = Nd4j.rand(1,inputs);

        w2 = Nd4j.rand(inputs,1);
        b2 = Nd4j.rand(1,1);


    }

    /**
     * 反向传播的训练过程
     * @param dataSet   数据集
     * 假如: 显示一个简单的2层网络
     *
     */
    public void train(DataSet dataSet){

        log.info("x:shape->{},y:shape->{},w1:shape->{}," +
                        "b1:shape->{},w2:shape->{},b2:shape->{}"
                ,dataSet.getX().shape(),dataSet.getY().shape(),w1.shape(),
                b1.shape(),w2.shape(),b2.shape());

        int m = dataSet.getX().rows();


        for(int i=0;i<batch;i++) {
            //FP
            INDArray z1 = dataSet.getX().mmul(w1).add(b1.broadcast(m,b1.columns()));
            INDArray a1 = Transforms.sigmoid(z1);

            INDArray z2 = a1.mmul(w2).add(b2.broadcast(m,b2.columns()));
            INDArray a2 = Transforms.sigmoid(z2);

            //BP
            delta2 = dataSet.getY().sub(a2);
            delta1 = delta2.mmul(w2.transpose()).mul(a1.mul(a1.rsub(1)));

            dw1 = dataSet.getX().transpose().mmul(delta1).add(w1.mul(lambda));
            db1 = Nd4j.sum(delta1, 0);
            dw2 = a1.transpose().mmul(delta2).add(w2.mul(lambda));
            db2 = Nd4j.sum(delta2, 0);

            w1 = w1.sub(dw1.mul(alpha));
            b1 = b1.sub(db1.mul(alpha));
            w2 = w2.sub(dw2.mul(alpha));
            b2 = b2.sub(db2.mul(alpha));
        }

        log.info("w1:->\n{}",w1);
        log.info("b1:->\n{}",b1);
        log.info("w2:->\n{}",w2);
        log.info("b2:->\n{}",b2);

    }





}
