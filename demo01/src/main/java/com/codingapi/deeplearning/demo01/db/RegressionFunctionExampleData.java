package com.codingapi.deeplearning.demo01.db;

import com.codingapi.deeplearning.demo01.domian.ExampleData;
import com.codingapi.deeplearning.demo01.learn.ExampleDataArrays;
import com.codingapi.deeplearning.demo01.learn.RegressionFunction;
import com.codingapi.deeplearning.demo01.mapper.ExampleDataMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.List;

/**
 * 数据操作工具类
 * @author lorne
 * @date 2019-10-22
 * @description y = 1+5x
 */
@Component
public class RegressionFunctionExampleData {

    @Autowired
    private ExampleDataMapper exampleDataMapper;


    public void randData(int number,double a,double b){
        exampleDataMapper.tuncate();
        RegressionFunction regressionFunction = new RegressionFunction(a,b);
        for(int i=0;i<number;i++){
            //为什么将x/number ,是为了梯度下降可更高效的操作，而做的特征缩放
            BigDecimal x = new BigDecimal(i).divide(new BigDecimal(number),5,BigDecimal.ROUND_UP);
            BigDecimal y =  regressionFunction.getY(x);
            exampleDataMapper.save(new ExampleData(x,y));
        }
    }


    public List<ExampleData> findAll(){
        return exampleDataMapper.findAll();
    }


    public ExampleDataArrays loadData(){
        List<ExampleData> exampleDataList = findAll();
        ExampleDataArrays exampleDataArrays = new ExampleDataArrays();
        int size = exampleDataList.size();
        double [] x = new double[size];
        double [] y = new double[size];
        for(int i=0;i<size;i++){
            ExampleData exampleData = exampleDataList.get(i);
            x[i] = exampleData.getX().doubleValue();
            y[i] = exampleData.getY().doubleValue();
        }
        exampleDataArrays.setX(x);
        exampleDataArrays.setY(y);
        return exampleDataArrays;
    }


}
