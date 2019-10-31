package com.codingapi.deeplearning.demo05;

import com.codingapi.deeplearning.demo05.learn.NeuralNetwork;
import com.codingapi.deeplearning.demo05.learn.DataSetScalingHelper;
import com.codingapi.deeplearning.demo05.learn.DataSet;
import com.codingapi.deeplearning.demo05.learn.SimpleNeuralNetworkLayerBuilder;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.IOException;

/**
 * @author lorne
 * @date 2019-10-31
 * @description
 */
@SpringBootTest
class DeepLearningJavaDemo05ApplicationTest {


    @Test
    void train() throws IOException {

        DataSet dataSet = new DataSet();
        DataSetScalingHelper scalingHelper = new DataSetScalingHelper(dataSet);
        scalingHelper.scaling();

        SimpleNeuralNetworkLayerBuilder simpleNeuralNetworkLayerBuilder
                = SimpleNeuralNetworkLayerBuilder.build()
                .addLayer(dataSet.inputSize(),5)
                .addLayer(5,3)
                .outLayer(3,1);

        NeuralNetwork neuralNetwork =
                new NeuralNetwork(0,0.1,100
                        ,simpleNeuralNetworkLayerBuilder);

        neuralNetwork.train(dataSet);

    }

}