package com.codingapi.deeplearning.demo10.learn.core;

import lombok.extern.slf4j.Slf4j;

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
    public void done(long index,double cost) {
        if(index % printIterations ==0) {
            log.info("index:{},cost:{}",index,cost);
        }
    }
}
