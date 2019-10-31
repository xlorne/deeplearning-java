package com.codingapi.deeplearning.demo05.learn;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.IOException;


/**
 * @author lorne
 * @date 2019-10-31
 * @description
 */
@SpringBootTest
class BackPropagationFunctionTest {


    @Test
    void train() throws IOException {
        DataSet dataSet = new DataSet();
        DataScalingHelper scalingHelper = new DataScalingHelper(dataSet);
        scalingHelper.scaling();

        BackPropagationFunction backPropagationFunction =
                new BackPropagationFunction(0,0.1,100,3);
        backPropagationFunction.train(dataSet);
    }
}