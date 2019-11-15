package com.codingapi.deeplearning.demo06.learn;

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
    private ScoreIterationListener iterationListener;

    public NeuralNetwork(double lambda, double alpha, int batch,
                         NeuralNetworkLayerBuilder builder) {
        this.lambda = lambda;
        this.alpha = alpha;
        this.batch = batch;
        this.builder = builder;

        //初始化权重
        builder.init();
    }


    public void addScoreIterationListener(ScoreIterationListener scoreIterationListener){
        this.iterationListener = scoreIterationListener;
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
            INDArray delta = data.sub(dataSet.getY());
            NeuralNetworkLayer outLayer = builder.get(builder.size()-1);
            delta = outLayer.back(delta,lambda);

            //倒数第2层开始向后传播
            for(int j=builder.size()-2;j>=0;j-- ){
                NeuralNetworkLayer layer = builder.get(j);
                layer = builder.get(j);
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
