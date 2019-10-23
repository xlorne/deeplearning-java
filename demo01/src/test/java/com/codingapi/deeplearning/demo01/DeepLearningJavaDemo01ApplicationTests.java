package com.codingapi.deeplearning.demo01;

import com.codingapi.deeplearning.demo01.learn.ExampleDataArrays;
import com.codingapi.deeplearning.demo01.learn.GradientDescentAlgorithmFunction;
import com.codingapi.deeplearning.demo01.db.RegressionFunctionExampleData;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
@Slf4j
class DeepLearningJavaDemo01ApplicationTests {

    @Autowired
    private RegressionFunctionExampleData regressionFunctionExampleData;


    @Test
    void randomData() {
        regressionFunctionExampleData.randData(100,1,2);
        log.info("data->{}",regressionFunctionExampleData.findAll());
    }


    @Test
    void train(){
        GradientDescentAlgorithmFunction gradientDescentAlgorithmFunction
                = new GradientDescentAlgorithmFunction(0.01,0.1,0.1,20000);
        ExampleDataArrays exampleDataArrays =  regressionFunctionExampleData.loadData();
        gradientDescentAlgorithmFunction.train(exampleDataArrays.getX(),exampleDataArrays.getY());
    }
}
