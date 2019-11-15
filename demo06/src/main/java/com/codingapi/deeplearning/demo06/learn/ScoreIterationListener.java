package com.codingapi.deeplearning.demo06.learn;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * 计算代价函数的值，打印得分
 * @author lorne
 * @date 2019-10-31
 * @description 逻辑回归损失函数
 */
@Slf4j
public class ScoreIterationListener {

    private int printIterations;

    private LossFunction lossFunction;

    private ScoreDoing scoreDoing;

    public ScoreIterationListener(int printIterations,ScoreDoing scoreDoing) {
        this.printIterations = printIterations;
        this.scoreDoing = scoreDoing;
    }

    public void init(LossFunction lossFunction){
        this.lossFunction = lossFunction;
    }

    public void cost(int index,INDArray predict, INDArray y) {
        if(index % printIterations ==0) {
            INDArray sum =  lossFunction.score(predict,y);
            scoreDoing.doing(sum);
        }
    }


    public static interface ScoreDoing{
        /**
         * 执行监听业务
         * @param sum   损失值
         */
        void doing(INDArray sum);
    }

}
