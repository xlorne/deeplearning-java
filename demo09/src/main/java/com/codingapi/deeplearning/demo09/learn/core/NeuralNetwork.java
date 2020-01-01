package com.codingapi.deeplearning.demo09.learn.core;

import com.codingapi.deeplearning.demo09.learn.layer.NeuralNetworkLayer;
import com.codingapi.deeplearning.demo09.learn.layer.NeuralNetworkLayerBuilder;
import com.codingapi.deeplearning.demo09.learn.loss.LossFunction;
import com.codingapi.deeplearning.demo09.learn.utils.MaxUtils;
import com.codingapi.deeplearning.demo09.learn.utils.SerializeUtils;
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


    protected NeuralNetwork(double lambda, double alpha, int numEpochs,long seed,
                         NeuralNetworkLayerBuilder layerBuilder,LossFunction lossFunction) {
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
    public void fit(DataSetIterator iterator){
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

    public int perdictIndex(INDArray data){
        return MaxUtils.maxIndex(predict(data).toDoubleVector());
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
