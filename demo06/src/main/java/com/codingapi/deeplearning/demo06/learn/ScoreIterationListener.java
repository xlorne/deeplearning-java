package com.codingapi.deeplearning.demo06.learn;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * 计算代价函数的值，打印得分
 * @author lorne
 * @date 2019-10-31
 * @description 逻辑回归损失函数
 */
@Slf4j
public class ScoreIterationListener {

    private int printIterations;

    public ScoreIterationListener(int printIterations) {
        this.printIterations = printIterations;
    }

    public void cost(int index,INDArray predict, INDArray y) {

        if(index % printIterations ==0) {

            INDArray first = y.mul(-1).mul(Transforms.log(predict));
            INDArray second = y.rsub(1).mul(Transforms.log(predict.rsub(1)));

            INDArray cost = first.sub(second);
            INDArray sum = Nd4j.sum(cost.div(y.rows()));
            log.info("index:{}=>cost:{}", index,sum.sumNumber().doubleValue());
        }
    }

}
