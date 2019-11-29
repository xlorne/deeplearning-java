package com.codingapi.deeplearning.demo06;

import com.codingapi.deeplearning.demo06.learn.*;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.IOException;

/**
 *
 * @author lorne
 * @date 2019-10-31
 */
@SpringBootTest
class DeepLearningJavaDemo06ApplicationTest {


    @Test
    void train() throws IOException {
        //创建数据集
        DataSet dataSet = new DataSet();
        //特征缩放
        DataSetScalingHelper scalingHelper = new DataSetScalingHelper(dataSet);
        scalingHelper.scalingSelf();

        //创建神经网络层
        NeuralNetworkLayerBuilder neuralNetworkLayerBuilder
                = NeuralNetworkLayerBuilder.Builder()
                .addLayer(DenseLayer.builder()
                        .input(dataSet.inputSize(),3)
                        .activation(new SigmoidActivation())
                        .build())
                .addLayer(DenseLayer.builder()
                        .input(3,2)
                        .activation(new SoftMaxActivation())
                        .isOutLayer(true)
                        .build())
                .builder();

        //创建神经网络
        NeuralNetwork neuralNetwork =
                 NeuralNetwork.builder()
                        .layers(neuralNetworkLayerBuilder)
                        .lossFunction(new SoftMaxLossFunction())
                        .seed(123)
                        .batchSize(13)
                        .numEpochs(20000)
                        .alpha(0.05)
                        .lambda(1e-5)
                        .build();

        //Loss函数监听
        neuralNetwork.initListeners(new ScoreLogTrainingListener(100));

        //训练数据
        neuralNetwork.train(dataSet);

        //预测数据
        INDArray test = Nd4j.create(1,2);
        test.putScalar(0,0,7.8);
        test.putScalar(0,1,4.3);

        //返回的百分比
        System.out.println("test:"+test);
        INDArray res = neuralNetwork.predict(scalingHelper.scaling(test));
        System.out.println("res的可能性结果:"+res);
    }

}