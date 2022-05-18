<template>
  <div>
    <el-divider content-position="left">基本参数</el-divider>
    <el-row :gutter="20" type="flex" class="row-bg" justify="center">
      <el-col :span="8">
        <el-form
          :model="ruleForm"
          status-icon
          :rules="rules"
          ref="ruleForm"
          label-width="100px"
          class="demo-ruleForm"
        >
          <el-form-item label="分类器" prop="epoch">
            <el-select v-model="classifier_value" placeholder="选择分类器">
              <el-option
                v-for="item in classifier_options"
                :key="item.value"
                :label="item.label"
                :value="item.value"
              >
              </el-option>
            </el-select>
          </el-form-item> </el-form
      ></el-col>

      <el-col :span="8">
        <el-form
          :model="ruleForm"
          status-icon
          :rules="rules"
          ref="ruleForm"
          label-width="100px"
          class="demo-ruleForm"
        >
          <el-form-item label="数据集" prop="dateset">
            <el-select v-model="dataset_value" placeholder="选择数据集">
              <el-option
                v-for="item in dataset_options"
                :key="item.value"
                :label="item.label"
                :value="item.value"
              >
              </el-option> </el-select
          ></el-form-item> </el-form
      ></el-col>
      <el-col :span="8" v-if="classifier_value == 'GNN'">
        <el-form
          :model="ruleForm"
          status-icon
          :rules="rules"
          label-width="100px"
        >
          <el-form-item label="threshold(%)" prop="threshold">
            <el-input v-model.number="ruleForm.threshold"></el-input>
          </el-form-item> </el-form
      ></el-col>
    </el-row>
    <el-divider content-position="left">模型参数</el-divider>
    <el-row :gutter="20" type="flex" class="row-bg" justify="center">
      <el-col :span="8"
        ><el-form
          :model="ruleForm"
          status-icon
          :rules="rules"
          ref="ruleForm"
          label-width="100px"
          class="demo-ruleForm"
        >
          <el-form-item label="epoch" prop="epoch">
            <el-input v-model.number="ruleForm.epoch"></el-input>
          </el-form-item> </el-form
      ></el-col>
      <el-col :span="8"
        ><el-form
          :model="ruleForm"
          status-icon
          :rules="rules"
          ref="ruleForm"
          label-width="100px"
          class="demo-ruleForm"
        >
          <el-form-item label="batchsize" prop="batchsize">
            <el-input v-model.number="ruleForm.batchsize"></el-input>
          </el-form-item> </el-form
      ></el-col>
      <el-col :span="8" v-if="classifier_value == 'GNN'">
        <el-form
          :model="ruleForm"
          status-icon
          :rules="rules"
          label-width="100px"
        >
          <el-form-item label="图分类方式" prop="graphClassifier">
            <el-select
              v-model="graph_classifier_value"
              placeholder="选择分类方式"
            >
              <el-option
                v-for="item in graph_classifier_options"
                :key="item.value"
                :label="item.label"
                :value="item.value"
              >
              </el-option> </el-select
          ></el-form-item> </el-form
      ></el-col>
    </el-row>
    <el-divider content-position="left">特征选择(百分比)</el-divider>
    <el-row :gutter="20" type="flex" class="row-bg" justify="center">
      <el-col :span="16">
        <el-slider v-model="sliderValue" show-input> </el-slider>
      </el-col>
    </el-row>
    <el-divider content-position="left">训练模型</el-divider>
    <el-row :gutter="20" type="flex" class="row-bg" justify="center">
      <el-col :span="6">
        <el-form
          :model="ruleForm"
          status-icon
          :rules="rules"
          ref="ruleForm"
          label-width="100px"
          class="demo-ruleForm"
        >
          <el-form-item>
            <el-button type="success" round @click="sendData"
              >开始评估</el-button
            >
          </el-form-item>
        </el-form></el-col
      >
    </el-row>
    <el-divider content-position="left" v-if="finished">结果查看</el-divider>
    <el-row
      :gutter="20"
      type="flex"
      class="row-bg"
      justify="center"
      v-if="finished"
    >
      <el-col :span="4">
        <el-card shadow="hover"> {{ accResult }} </el-card>
      </el-col>
      <el-col :span="4">
        <el-card shadow="hover"> {{ aucResult }}</el-card>
      </el-col>
    </el-row>
  </div>
</template>
<script>
import axios from "axios";

export default {
  data() {
    var checkNumber = (rule, value, callback) => {
      if (!value) {
        return callback(new Error("不能为空"));
      }
      setTimeout(() => {
        if (!Number.isInteger(value)) {
          callback(new Error("请输入数字值"));
        } else {
          if (value < 0) {
            callback(new Error("必须为正数"));
          } else {
            callback();
          }
        }
      }, 500);
    };
    var checkThresh = (rule, value, callback) => {
      if (!value) {
        return callback(new Error("不能为空"));
      }
      setTimeout(() => {
        if (!Number.isFinite(value)) {
          callback(new Error("请输入百分比"));
        } else {
          if (value < 1 || value > 100) {
            callback(new Error("必须在1~100之间"));
          } else {
            callback();
          }
        }
      }, 500);
    };
    return {
      finished: false,

      sliderValue: 0,
      switchValue: true,
      accResult: "",
      aucResult: "",
      ruleForm: {
        batchsize: "",
        epoch: "",
        threshold: "",
      },
      rules: {
        batchsize: [{ validator: checkNumber, trigger: "blur" }],
        epoch: [{ validator: checkNumber, trigger: "blur" }],
        threshold: [{ validator: checkThresh, trigger: "blur" }],
      },
      graph_classifier_value: "",
      graph_classifier_options: [
        {
          value: "DiffPool",
          label: "DiffPool",
        },
        {
          value: "GlobalMean",
          label: "GlobalMean",
        },
      ],
      classifier_options: [
        {
          value: "SVM",
          label: "SVM",
        },
        {
          value: "RF",
          label: "RF",
        },
        {
          value: "GNN",
          label: "GNN",
        },
        {
          value: "1D-CNN",
          label: "1D-CNN",
        },
        {
          value: "DNN",
          label: "DNN",
        },
      ],
      //   value值是被选中的值
      classifier_value: "",
      dataset_options: [
        {
          value: "cirrhosis",
          label: "cirrhosis",
        },
        {
          value: "ibd",
          label: "ibd",
        },
        {
          value: "colorectal",
          label: "colorectal",
        },
        {
          value: "obesity",
          label: "obesity",
        },
        {
          value: "t2d",
          label: "t2d",
        },
        {
          value: "wt2d",
          label: "wt2d",
        },
      ],
      //   value值是被选中的值
      dataset_value: "",
    };
  },
  methods: {
    getData() {
      var that = this;
      // 对应 Python 提供的接口，这里的地址填写下面服务器运行的地址，本地则为127.0.0.1，外网则为 your_ip_address
      const path = "http://127.0.0.1:5000/DieaseDiagnosize";
      axios
        .get(path)
        .then(function (response) {
          // 这里服务器返回的 response 为一个 json object，可通过如下方法需要转成 json 字符串
          // 可以直接通过 response.data 取key-value
          // 坑一：这里不能直接使用 this 指针，不然找不到对象
          var record = response.data;
          // 坑二：这里直接按类型解析，若再通过 JSON.stringify(msg) 转，会得到带双引号的字串
          // alert(
          //   "Success " + response.status + ", " + response.data + ", " + record
          // );
          that.record = record;
        })
        .catch(function (error) {
          alert("Error " + error);
        });
    },
    sendData() {
      if (
        this.classifier_value == "" ||
        this.dataset_value == "" ||
        this.ruleForm.batchsize == 0 ||
        this.ruleForm.epoch == 0 ||
        this.ruleForm.threshold == 0
      ) {
        console.log(this.ruleForm.threshold);
        alert("value cannot be null");
        return;
      }
      var that = this;
      // 对应 Python 提供的接口，这里的地址填写下面服务器运行的地址，本地则为127.0.0.1，外网则为 your_ip_address
      const path = "http://127.0.0.1:5000/ModelEvaluation";
      axios
        .post(path, {
          dataset: that.dataset_value,
          classifier: that.classifier_value,
          feature: that.sliderValue,
          graph_classifier: that.graph_classifier_value,
          epoch: that.ruleForm.epoch,
          batchsize: that.ruleForm.batchsize,
          threshold: that.ruleForm.threshold,
        })
        .then(function (response) {
          // 这里服务器返回的 response 为一个 json object，可通过如下方法需要转成 json 字符串
          // 可以直接通过 response.data 取key-value
          // 坑一：这里不能直接使用 this 指针，不然找不到对象
          let finished = response.data.finished;
          let acc = response.data.acc;
          let auc = response.data.auc;
          // 坑二：这里直接按类型解析，若再通过 JSON.stringify(msg) 转，会得到带双引号的字串
          // alert(
          //   "Success " + response.status + ", " + response.data + ", " + record
          // );
          console.log(acc, auc);
          that.finished = finished;
          that.accResult = "acc: " + acc;
          that.aucResult = "auc: " + auc;
        })
        .catch(function (error) {
          alert("Error " + error);
        });
    },
  },
  watch: {
    classifier_value(val) {
      if (val != "GNN") {
        this.ruleForm.threshold = "_";
        this.graph_classifier_value = "_";
        console.log(val, this.graph_classifier_value + this.ruleForm.threshold);
      } else {
        this.ruleForm.threshold = "40";
        this.graph_classifier_value = "DiffPool";
        console.log(this.graph_classifier_value + this.ruleForm.threshold);
      }
    },
  },
  mounted() {
    this.getData();
    this.threshold = "33";
  },
};
</script>
<style>
.row-bg {
  width: 80%;
}
.el-row {
  margin-left: 20px;
  margin-right: 20px;
}
.el-divider {
  background-color: #c8cdd8;
}
.el-card {
  background-color: #eee;
}
.el-divider__text {
  background-color: #eee;
  font-size: 20px;
}
</style>