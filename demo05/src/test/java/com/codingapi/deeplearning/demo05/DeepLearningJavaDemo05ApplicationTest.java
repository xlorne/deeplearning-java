package com.codingapi.deeplearning.demo05;

import com.codingapi.deeplearning.demo05.learn.*;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
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
        scalingHelper.scalingSelf();

        //创建神经网络层
        SimpleNeuralNetworkLayerBuilder simpleNeuralNetworkLayerBuilder
                = SimpleNeuralNetworkLayerBuilder.build()
                .addLayer(dataSet.inputSize(),2)
                .outLayer(2,1);

        //创建神经网络
        NeuralNetwork neuralNetwork =
                new NeuralNetwork(0,0.3,20000
                        ,simpleNeuralNetworkLayerBuilder);

        //代价函数得分
        neuralNetwork.addScoreIterationListener(new ScoreIterationListener(1000));

        //训练数据
        neuralNetwork.train(dataSet);

        //预测数据
        INDArray test = Nd4j.create(1,2);
        test.putScalar(0,0,7.5);
        test.putScalar(0,1,4.5);

        //返回的百分比
        System.out.println("test:"+test);
        INDArray res = neuralNetwork.predict(scalingHelper.scaling(test));
        System.out.println("res为1的可能性为:"+res);
    }

}