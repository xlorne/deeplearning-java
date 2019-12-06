package com.codingapi.deeplearning.demo07;

import com.codingapi.deeplearning.demo07.learn.*;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.IOException;

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
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        //创建神经网络层
        NeuralNetworkLayerBuilder neuralNetworkLayerBuilder
                = NeuralNetworkLayerBuilder.builder()
                .addLayer(DenseLayer.builder()
                        .input(28*28,1000)
                        .activation(new SigmoidActivation())
                        .build())
                .addLayer(DenseLayer.builder()
                        .input(1000,1000)
                        .activation(new SigmoidActivation())
                        .build())
                .addLayer(DenseLayer.builder()
                        .input(1000,10)
                        .activation(new SoftMaxActivation())
                        .isOutLayer(true)
                        .build())
                .build();

        //创建神经网络
        NeuralNetwork neuralNetwork =
                 NeuralNetwork.builder()
                        .layers(neuralNetworkLayerBuilder)
                        .lossFunction(new SoftMaxLossFunction())
                        .seed(rngSeed)
                        .numEpochs(1)
                        .alpha(0.006)
                        .lambda(1e-3)
                        .build();

        //Loss函数监听
        neuralNetwork.initListeners(new ScoreLogTrainingListener(1));

        //训练数据
        neuralNetwork.train(mnistTrain);

        //预测数据
        double count = 0;
        int success = 0;
        while (mnistTest.hasNext()){
            DataSet dataSet =  mnistTest.next();
            INDArray res = neuralNetwork.predict(dataSet.getFeatures());
            INDArray labels = dataSet.getLabels();
            int rows = res.rows();
            for(int i=0;i<rows;i++){
                count++;
                if(MaxUtils.maxIndex(res.getRow(i).toDoubleVector())==MaxUtils.maxIndex(labels.getRow(i).toDoubleVector())){
                    success++;
                }
            }
        }
        System.out.println(String.format("success:%f",(success/count)));


    }

}