package com.codingapi.deeplearning.demo07;

import com.codingapi.deeplearning.demo07.learn.*;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.IOException;
import java.util.Arrays;

/**
 *
 * @author lorne
 * @date 2019-10-31
 */
@SpringBootTest
class DeepLearningJavaDemo07ApplicationTest {


    @Test
    void train() throws IOException {

        int batchSize = 64;
        int rngSeed = 123;

        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, true, rngSeed);


        //创建神经网络层
        NeuralNetworkLayerBuilder neuralNetworkLayerBuilder
                = NeuralNetworkLayerBuilder.Builder()
                .addLayer(DenseLayer.builder()
                        .input(28*28,1000)
                        .activation(new SigmoidActivation())
                        .build())
                .addLayer(DenseLayer.builder()
                        .input(1000,10)
                        .activation(new SoftMaxActivation())
                        .isOutLayer(true)
                        .build())
                .builder();

        //创建神经网络
        NeuralNetwork neuralNetwork =
                 NeuralNetwork.builder()
                        .layers(neuralNetworkLayerBuilder)
                        .lossFunction(new SoftMaxLossFunction())
                        .seed(rngSeed)
                        .numEpochs(15)
                        .alpha(0.005)
                        .lambda(1e-4)
                        .build();

        //Loss函数监听
        neuralNetwork.initListeners(new ScoreLogTrainingListener(1));

        //训练数据
        neuralNetwork.train(mnistTrain);

        //预测数据
        while (mnistTest.hasNext()){
            DataSet dataSet =  mnistTest.next();
            INDArray res = neuralNetwork.predict(dataSet.getFeatures());
            INDArray labels = dataSet.getLabels();
            if(Arrays.equals(res.toFloatVector(),labels.toFloatVector())){
                System.out.println("res的可能性结果:Ok");
            }else{
                System.out.println("res的可能性结果:Error");
            }
        }


    }

}