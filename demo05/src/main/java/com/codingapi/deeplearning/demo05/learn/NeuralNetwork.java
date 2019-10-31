package com.codingapi.deeplearning.demo05.learn;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author lorne
 * @date 2019-10-31
 * @description 神经网络实现
 */
@Slf4j
public class NeuralNetwork {


    private double lambda;
    private double alpha;

    private int batch;

    private SimpleNeuralNetworkLayerBuilder builder;

    public NeuralNetwork(double lambda, double alpha, int batch,
                         SimpleNeuralNetworkLayerBuilder builder) {
        this.lambda = lambda;
        this.alpha = alpha;
        this.batch = batch;
        this.builder = builder;
    }


    /**
     * 反向传播的训练过程
     * @param dataSet   数据集
     *
     */
    public void train(DataSet dataSet){

        for(int i=0;i<batch;i++) {
            //向前传播算法 FP
            INDArray data = dataSet.getX();
            for(int j=0;j<builder.size();j++ ){
                SimpleNeuralNetworkLayer layer = builder.get(j);
                data = layer.forward(data);
            }

            //反向传播 BP
            //输出层的反向传播
            INDArray delta = dataSet.getY().sub(data);
            SimpleNeuralNetworkLayer outLayer = builder.get(builder.size()-1);
            delta = outLayer.back(delta,lambda);

            //倒数第2层开始向后传播
            for(int j=builder.size()-2;j>=0;j-- ){
                SimpleNeuralNetworkLayer layer = builder.get(j);
                layer = builder.get(j);
                delta = layer.back(delta,lambda);
            }

            for(int j=0;j<builder.size();j++ ){
                SimpleNeuralNetworkLayer layer = builder.get(j);
                layer.updateParam(alpha);
            }
        }

    }

}
