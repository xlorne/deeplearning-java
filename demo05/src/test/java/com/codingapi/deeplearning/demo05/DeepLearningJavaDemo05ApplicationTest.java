package com.codingapi.deeplearning.demo05;

import com.codingapi.deeplearning.demo05.learn.*;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.IOException;

/**
 *
 * @author lorne
 * @date 2019-10-31
 * @description
 */
@SpringBootTest
class DeepLearningJavaDemo05ApplicationTest {


    @Test
    void train() throws IOException {

        //创建数据集
        DataSet dataSet = new DataSet();
        //特征缩放
        DataSetScalingHelper scalingHelper = new DataSetScalingHelper(dataSet);
        scalingHelper.scaling();

        //创建神经网络层
        SimpleNeuralNetworkLayerBuilder simpleNeuralNetworkLayerBuilder
                = SimpleNeuralNetworkLayerBuilder.build()
                .addLayer(dataSet.inputSize(),3)
                .addLayer(3,2)
                .outLayer(2,1);

        //创建神经网络
        NeuralNetwork neuralNetwork =
                new NeuralNetwork(1e-4,0.1,10000
                        ,simpleNeuralNetworkLayerBuilder);

        //代价函数得分
        neuralNetwork.addScoreIterationListener(new ScoreIterationListener(100));

        //训练数据
        neuralNetwork.train(dataSet);
    }

}