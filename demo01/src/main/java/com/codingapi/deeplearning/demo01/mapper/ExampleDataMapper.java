package com.codingapi.deeplearning.demo01.mapper;

import com.codingapi.deeplearning.demo01.domian.ExampleData;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.util.List;

/**
 * @author lorne
 * @date 2019-10-22
 * @description
 */
@Mapper
public interface ExampleDataMapper {

    @Insert("insert into t_example_data(x,y) values(#{x},#{y})")
    int save(ExampleData exampleData);

    @Select("select * from t_example_data")
    List<ExampleData> findAll();

    @Update("TRUNCATE t_example_data")
    int tuncate();

}
