package com.codingapi.deeplearning.demo03;

import com.codingapi.deeplearning.demo03.db.RegressionFunctionExampleData;
import com.codingapi.deeplearning.demo03.learn.GradientDescentAlgorithmFunction;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
@Slf4j
class DeepLearningJavaDemo03ApplicationTests {

    @Autowired
    private RegressionFunctionExampleData regressionFunctionExampleData;


    @Test
    void randomData() {
        regressionFunctionExampleData.randData(100,1,2);
    }


    @Test
    void train(){
        GradientDescentAlgorithmFunction gradientDescentAlgorithmFunction
                = new GradientDescentAlgorithmFunction(0.01,10000);
        INDArray exampleDataArrays =  regressionFunctionExampleData.loadData();
        gradientDescentAlgorithmFunction.train(exampleDataArrays);
    }
}
