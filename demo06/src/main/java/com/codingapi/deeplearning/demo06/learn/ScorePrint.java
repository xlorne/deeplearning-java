package com.codingapi.deeplearning.demo06.learn;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author lorne
 * @date 2019-11-15
 * @description
 */
@Slf4j
public class ScorePrint implements ScoreIterationListener.ScoreDoing {

    @Override
    public void doing(INDArray sum) {
      log.info("cost:{}",sum.amaxNumber().doubleValue());
    }
}
