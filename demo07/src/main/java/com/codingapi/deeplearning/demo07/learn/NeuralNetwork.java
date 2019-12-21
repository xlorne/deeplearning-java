package com.codingapi.deeplearning.demo07.learn;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
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
    private int numEpochs;

    /**
     * 神经网络层
     */
    private NeuralNetworkLayerBuilder layerBuilder;

    /**
     * 监听函数
     */
    private NeuralListener iterationListener;

    /**
     * 损失函数
     */
    private LossFunction lossFunction;


    public static NeuralNetworkBuilder builder(){
        return new NeuralNetworkBuilder();
    }

    public static class NeuralNetworkBuilder{
        private NeuralNetworkLayerBuilder builder;
        private double lambda;
        private double alpha;
        private int numEpochs;
        private long seed;
        private LossFunction lossFunction;

        private NeuralNetworkBuilder() {
            lambda = 0;
            alpha = 0.1;
            numEpochs = 10000;
            seed = 123;
        }

        public NeuralNetworkBuilder layers(NeuralNetworkLayerBuilder builder){
            this.builder = builder;
            return this;
        }

        public NeuralNetworkBuilder numEpochs(int numEpochs){
            this.numEpochs = numEpochs;
            return this;
        }

        public NeuralNetworkBuilder lambda(double lambda){
            this.lambda = lambda;
            return this;
        }

        public NeuralNetworkBuilder alpha(double alpha){
            this.alpha = alpha;
            return this;
        }

        public NeuralNetworkBuilder seed(long seed){
            this.seed = seed;
            return this;
        }

        public NeuralNetworkBuilder lossFunction(LossFunction lossFunction){
            this.lossFunction = lossFunction;
            return this;
        }

        public NeuralNetwork build(){
            return new NeuralNetwork(lambda,alpha,numEpochs,seed,builder,lossFunction);
        }


    }

    private NeuralNetwork(double lambda, double alpha, int numEpochs,long seed,
                         NeuralNetworkLayerBuilder layerBuilder,LossFunction lossFunction) {
        this.lambda = lambda;
        this.alpha = alpha;
        this.numEpochs = numEpochs;
        this.layerBuilder = layerBuilder;
        this.lossFunction = lossFunction;
        Nd4j.getRandom().setSeed(seed);
        //初始化权重
        layerBuilder.init(lambda,alpha,seed);
    }


    public void initListeners(NeuralListener.TrainingListener... trainingListeners){
        this.iterationListener = new NeuralListener(trainingListeners);
        this.iterationListener.init(lossFunction);
    }

    /**
     * 训练过程
     * @param iterator   数据集
     *
     */
    public void train(DataSetIterator iterator){
        log.info("train => start");
        long count = 0;
        for(int i=1;i<=numEpochs;i++) {
            //向前传播算法 FP
            while (iterator.hasNext()) {
                DataSet batch = iterator.next();
                INDArray data = batch.getFeatures();
                INDArray label = batch.getLabels();
                for (int j = 0; j < layerBuilder.size(); j++) {
                    NeuralNetworkLayer layer = layerBuilder.get(j);
                    data = layer.forward(data);
                }

                //反向传播 BP
                //输出层的反向传播
                INDArray delta = lossFunction.gradient(data, label);

                for (int j = layerBuilder.size() - 1; j >= 0; j--) {
                    NeuralNetworkLayer layer = layerBuilder.get(j);
                    delta = layer.backprop(delta);
                }

                //更新参数
                for (int j = 0; j < layerBuilder.size(); j++) {
                    NeuralNetworkLayer layer = layerBuilder.get(j);
                    layer.updateParam();
                }

                //损失函数得分
                if (iterationListener != null) {
                    iterationListener.cost(count++, data, label);
                }
            }
            iterator.reset();
        }
        log.info("train => over");

    }


    /**
     * 预测数据 返回的是100%
     * @param data  测试数据
     * @return  预测值
     */
    public INDArray predict(INDArray data){
        for(int j=0;j<layerBuilder.size();j++ ){
            NeuralNetworkLayer layer = layerBuilder.get(j);
            data = layer.forward(data);
        }
        return data;
    }


}
