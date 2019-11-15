package com.codingapi.deeplearning.demo06.learn;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author lorne
 * @date 2019-11-15
 * @description
 */
@Slf4j
public class ScoreLogTrainingListener implements NeuralListener.TrainingListener {

    private int printIterations;

    public ScoreLogTrainingListener(int printIterations) {
        this.printIterations = printIterations;
    }

    @Override
    public void done(int index,INDArray sum) {
        if(index % printIterations ==0) {
            log.info("cost:{}",sum.amaxNumber().doubleValue());
        }
    }
}
