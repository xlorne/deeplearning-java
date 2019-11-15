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
class DeepLearningJavaDemo05ApplicationTest {


    @Test
    void train() throws IOException {
        //创建数据集
        DataSet dataSet = new DataSet();
        //特征缩放
        DataSetScalingHelper scalingHelper = new DataSetScalingHelper(dataSet);
        scalingHelper.scalingSelf();

        //创建神经网络层
        NeuralNetworkLayerBuilder neuralNetworkLayerBuilder
                = NeuralNetworkLayerBuilder.build()
                .addLayer(new DenseLayer(dataSet.inputSize(),2))
                .addLayer(new DenseLayer(2,1,true));

        //创建神经网络
        NeuralNetwork neuralNetwork =
                new NeuralNetwork(0,0.1,20000
                        , neuralNetworkLayerBuilder,new LogisticRegressionLossFunction());

        //Loss函数监听
        neuralNetwork.initScoreIterationListener(1000,new ScorePrint());

        //训练数据
        neuralNetwork.train(dataSet);

        //预测数据
        INDArray test = Nd4j.create(1,2);
        test.putScalar(0,0,7.8);
        test.putScalar(0,1,4.3);

        //返回的百分比
        System.out.println("test:"+test);
        INDArray res = neuralNetwork.predict(scalingHelper.scaling(test));
        System.out.println("res为1的可能性为:"+res);
    }

}