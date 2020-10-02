package com.codingapi.deeplearning.demo10.learn.core;

import com.codingapi.deeplearning.demo10.learn.layer.FeedForwardLayer;
import com.codingapi.deeplearning.demo10.learn.layer.NeuralNetworkLayer;
import com.codingapi.deeplearning.demo10.learn.layer.NeuralNetworkLayerBuilder;
import com.codingapi.deeplearning.demo10.learn.loss.LossFunction;
import com.codingapi.deeplearning.demo10.learn.utils.SerializeUtils;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;

/**
 *
 * @author lorne
 * @date 2019-10-31
 * @description 神经网络实现
 */
@Slf4j
public class NeuralNetwork implements Serializable {


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


    protected NeuralNetwork(double lambda, double alpha, int numEpochs, int seed,
                            NeuralNetworkLayerBuilder layerBuilder, LossFunction lossFunction, InputType inputType) {
        this.numEpochs = numEpochs;
        this.layerBuilder = layerBuilder;
        this.lossFunction = lossFunction;
        Nd4j.getRandom().setSeed(seed);
        //初始化权重
        layerBuilder.init(inputType,lambda,alpha,seed);
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
    public void fit(DataSetIterator iterator){
        log.info("train => start");
        long count = 0;
        for(int i=1;i<=numEpochs;i++) {
            //向前传播算法 FP
            while (iterator.hasNext()) {
                DataSet batch = iterator.next();
                INDArray data = batch.getFeatures();
                INDArray label = batch.getLabels();

                NeuralNetworkLayerBuilder.NeuralNetworkLayerIterator neuralNetworkLayerIterator
                        =  layerBuilder.neuralNetworkLayerIterator();
                while (neuralNetworkLayerIterator.hasNext()){
                    NeuralNetworkLayer layer = neuralNetworkLayerIterator.next();
                    data = layer.forward(data);
                }

                double cost =  lossFunction.score(data,label);

                //损失函数得分
                if (iterationListener != null) {
                    iterationListener.cost(count++, cost);
                }

                //反向传播 BP
                //输出层的反向传播
                INDArray delta = lossFunction.gradient(data, label);
                NeuralNetworkLayerBuilder.BFIterator bfIterator = layerBuilder.bFIterator();
                while (bfIterator.hasNext()){
                    FeedForwardLayer layer = bfIterator.next();
                    delta = layer.backprop(delta);
                }

                //更新参数
                NeuralNetworkLayerBuilder.FeedForwardLayerIterator feedForwardLayerIterator
                        =  layerBuilder.feedForwardLayerIterator();
                while (feedForwardLayerIterator.hasNext()){
                    FeedForwardLayer layer = feedForwardLayerIterator.next();
                    layer.updateParam();
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
        NeuralNetworkLayerBuilder.NeuralNetworkLayerIterator neuralNetworkLayerIterator =  layerBuilder.neuralNetworkLayerIterator();
        while (neuralNetworkLayerIterator.hasNext()){
            NeuralNetworkLayer layer = neuralNetworkLayerIterator.next();
            data = layer.forward(data);
        }
        return data;
    }


    /**
     * 将训练模型保存到本地
     * @param path  保存路径
     */
    public void save(String path) throws IOException {
       File file = new File(path);
       if(!file.exists()){
           file.createNewFile();
       }
       byte[] data = SerializeUtils.serialize(this);
       FileUtils.writeByteArrayToFile(file,data);
    }



}
