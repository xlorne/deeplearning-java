package com.codingapi.deeplearning.demo06.learn;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author lorne
 * @date 2019-10-31
 * @description 神经网络实现
 */
@Slf4j
public class NeuralNetwork {


    /**
     * 正则化参数
     */
    private double lambda;
    /**
     * 学习率
     */
    private double alpha;

    /**
     * 训练次数
     */
    private int batch;

    /**
     * 神经网络层
     */
    private NeuralNetworkLayerBuilder builder;

    /**
     * 监听函数
     */
    private NeuralListener iterationListener;

    /**
     * 损失函数
     */
    private LossFunction lossFunction;



    public static class Builder{
        private NeuralNetworkLayerBuilder builder;
        private double lambda;
        private double alpha;
        private int batch;
        private long seed;
        private LossFunction lossFunction;

        public Builder() {
            lambda = 0;
            alpha = 0.1;
            batch = 10000;
            seed = 123;
        }

        public Builder layers(NeuralNetworkLayerBuilder builder){
            this.builder = builder;
            return this;
        }

        public Builder batch(int batch){
            this.batch = batch;
            return this;
        }

        public Builder lambda(double lambda){
            this.lambda = lambda;
            return this;
        }

        public Builder alpha(double alpha){
            this.alpha = alpha;
            return this;
        }

        public Builder seed(long seed){
            this.seed = seed;
            return this;
        }

        public Builder lossFunction(LossFunction lossFunction){
            this.lossFunction = lossFunction;
            return this;
        }

        public NeuralNetwork build(){
            return new NeuralNetwork(lambda,alpha,batch,seed,builder,lossFunction);
        }


    }

    private NeuralNetwork(double lambda, double alpha, int batch,long seed,
                         NeuralNetworkLayerBuilder builder,LossFunction lossFunction) {
        this.lambda = lambda;
        this.alpha = alpha;
        this.batch = batch;
        this.builder = builder;
        this.lossFunction = lossFunction;
        Nd4j.getRandom().setSeed(seed);
        //初始化权重
        builder.init();
    }


    public void initListeners(NeuralListener.TrainingListener... trainingListeners){
        this.iterationListener = new NeuralListener(trainingListeners);
        this.iterationListener.init(lossFunction);
    }

    /**
     * 训练过程
     * @param dataSet   数据集
     *
     */
    public void train(DataSet dataSet){
        log.info("train => start");
        for(int i=1;i<=batch;i++) {
            //向前传播算法 FP
            INDArray data = dataSet.getX();
            for(int j=0;j<builder.size();j++ ){
                NeuralNetworkLayer layer = builder.get(j);
                data = layer.forward(data);
            }

            //反向传播 BP
            //输出层的反向传播
            INDArray delta = lossFunction.gradient(data,dataSet.getY());

            for(int j=builder.size()-1;j>=0;j-- ){
                NeuralNetworkLayer layer = builder.get(j);
                delta = layer.back(delta,lambda);
            }

            //更新参数
            for(int j=0;j<builder.size();j++ ){
                NeuralNetworkLayer layer = builder.get(j);
                layer.updateParam(alpha);
            }

            //损失函数得分
            if(iterationListener!=null){
                iterationListener.cost(i,data,dataSet.getY());
            }
        }
        log.info("train => over");

    }


    /**
     * 预测数据 返回的是100%
     * @param data  测试数据
     * @return  预测值
     */
    public INDArray predict(INDArray data){
        for(int j=0;j<builder.size();j++ ){
            NeuralNetworkLayer layer = builder.get(j);
            data = layer.forward(data);
        }
        return data;
    }


}
