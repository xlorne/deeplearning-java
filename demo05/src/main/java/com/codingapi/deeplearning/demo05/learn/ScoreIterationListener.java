package com.codingapi.deeplearning.demo05.learn;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * 计算代价函数的值，打印得分
 * @author lorne
 * @date 2019-10-31
 * @description
 */
@Slf4j
public class ScoreIterationListener {



    public void cost(INDArray prediction, INDArray y) {

        INDArray first = y.mul(-1).mul(Transforms.log(prediction));
        INDArray second = y.rsub(1).mul(Transforms.log(prediction.rsub(1)));

        INDArray cost = first.sub(second);
        INDArray sum = Nd4j.sum(cost.div(y.rows()));
        log.info("cost:->{}",sum.sumNumber().doubleValue());

    }
}
