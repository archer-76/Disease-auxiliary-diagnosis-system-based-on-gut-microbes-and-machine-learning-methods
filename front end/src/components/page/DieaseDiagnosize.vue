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
    <el-row
      :gutter="20"
      type="flex"
      class="row-bg"
      justify="center"
      v-if="classifier_value == 'RF'"
    >
      <el-col :span="8"
        ><el-form
          :model="ruleForm"
          status-icon
          :rules="rules"
          ref="ruleForm"
          label-width="100px"
          class="demo-ruleForm"
        >
          <el-form-item label="criterion:" prop="criterion">
            <el-input v-model.number="RfCriterion" :disabled="true"></el-input>
          </el-form-item> </el-form
      ></el-col>
      <el-col :span="8"
        ><el-form
          :model="ruleForm"
          status-icon
          :rules="rules"
          ref="ruleForm"
          label-width="150px"
          class="demo-ruleForm"
        >
          <el-form-item label="min_samples_split:" prop="min_samples_split">
            <el-input v-model.number="RfStop" :disabled="true"></el-input>
          </el-form-item> </el-form
      ></el-col>
    </el-row>
    <el-row
      :gutter="20"
      type="flex"
      class="row-bg"
      justify="center"
      v-if="classifier_value == 'SVM'"
    >
      <el-col :span="8"
        ><el-form
          :model="ruleForm"
          status-icon
          :rules="rules"
          ref="ruleForm"
          label-width="100px"
          class="demo-ruleForm"
        >
          <el-form-item label="gamma:" prop="gamma">
            <el-input v-model.number="SvmGamma" :disabled="true"></el-input>
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
          <el-form-item label="C:" prop="C">
            <el-input v-model.number="SvmC" :disabled="true"></el-input>
          </el-form-item> </el-form
      ></el-col>
    </el-row>
    <el-row
      :gutter="20"
      type="flex"
      class="row-bg"
      justify="center"
      v-if="classifier_value != 'RF' && classifier_value != 'SVM'"
    >
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
    </el-row>

    <el-divider content-position="left">特征选择(百分比)</el-divider>
    <el-row :gutter="20" type="flex" class="row-bg" justify="center">
      <el-col :span="16">
        <el-slider v-model="sliderValue" show-input> </el-slider>
      </el-col>
    </el-row>
    <el-divider content-position="left">疾病诊断</el-divider>
    <el-row :gutter="20" type="flex" class="row-bg" justify="center">
      <el-col :span="8">
        <el-form ref="form" :model="form" label-width="120px" style=" 50%">
          <el-form-item label="输入OTU表">
            <el-upload
              class="upload-demo"
              action=""
              :http-request="uploadFile"
              :limit="1"
              :on-exceed="handleExceed"
            >
              <el-button size="small" type="primary">点击上传</el-button>
            </el-upload>
          </el-form-item></el-form
        >
      </el-col>
      <el-col :span="8">
        <el-form
          :model="ruleForm"
          status-icon
          :rules="rules"
          ref="ruleForm"
          label-width="100px"
          class="demo-ruleForm"
        >
          <el-form-item label="启用预置模型" prop="premodel">
            <el-select v-model="premodel_value" placeholder="选择数据集">
              <el-option
                v-for="item in premodel_options"
                :key="item.value"
                :label="item.label"
                :value="item.value"
              >
              </el-option> </el-select
          ></el-form-item>
        </el-form>
      </el-col>
      <el-col :span="4">
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
              >开始诊断</el-button
            >
          </el-form-item>
        </el-form></el-col
      >
    </el-row>
    <el-row
      v-if="!notprocessing"
      :gutter="20"
      type="flex"
      class="row-bg"
      justify="center"
    >
      <el-col :span="8">
        <el-card shadow="hover"> 正在诊断疾病，时间漫长，请您稍等</el-card>
      </el-col>
    </el-row>
    <el-divider v-if="finished && notprocessing" content-position="left"
      >结果查看</el-divider
    >
    <el-row
      v-if="finished && notprocessing"
      :gutter="20"
      type="flex"
      class="row-bg"
      justify="center"
    >
      <el-col :span="8">
        <el-card shadow="hover">诊断结束，请前往诊断结果界面查看详情</el-card>
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
      SvmGamma: 1,
      SvmC: 10,
      RfCriterion: "基尼不纯度",
      RfStop: "2",
      record: [],
      form: {
        imgSavePath: "",
      },
      fileList: [
        {
          name: "food.jpeg",
          url: "https://fuss10.elemecdn.com/3/63/4e7f3a15429bfda99bce42a18cdd1jpeg.jpeg?imageMogr2/thumbnail/360x360/format/webp/quality/100",
        },
        {
          name: "food2.jpeg",
          url: "https://fuss10.elemecdn.com/3/63/4e7f3a15429bfda99bce42a18cdd1jpeg.jpeg?imageMogr2/thumbnail/360x360/format/webp/quality/100",
        },
      ],

      sliderValue: 0,
      switchValue: true,
      accResult: "acc:",
      aucResult: "auc:",
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
      premodel_value: "",
      premodel_options: [
        {
          value: "Nothing",
          label: "不启用",
        },
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
      notprocessing: true,
      staticFiles: [],
      mscUrl: "",
      mscName: "",
      epoch: "",
      batchsize: "",
      threshold: "",
      finished: false,
    };
  },
  methods: {
    uploadFile(param) {
      let fileObj = param.file;
      let form = new FormData();
      form.append("fileToUpload", fileObj);
      // form.append("diease", disease);
      console.log(form); // output is: FormData {}; 需要使用 .get() 来读取
      console.log(form.get("fileToUpload")); // output is exactly the fileObj

      axios.post("http://127.0.0.1:5000/HandleFileRequest", form, {
        headers: { "content-type": "multipart/form-data" },
      });
    },
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
          var record = response.data.record;
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
      var that = this;
      this.notprocessing = false;
      // 对应 Python 提供的接口，这里的地址填写下面服务器运行的地址，本地则为127.0.0.1，外网则为 your_ip_address
      var path = "http://127.0.0.1:5000/DieaseDiagnosize";
      if (this.classifier_value == "1D-CNN") {
        path = "http://127.0.0.1:5555/DieaseDiagnosize";
      }
      axios
        .post(path, {
          dataset: that.dataset_value,
          classifier: that.classifier_value,
          feature: that.sliderValue,
          pooling: that.graph_classifier_value,
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
          that.finished = finished;
          that.notprocessing = true;
          that.accResult = acc;
          that.aucResult = auc;
        })
        .catch(function (error) {
          alert("本次诊断结果可能出现异常，请勿在较小数据集上使用新颖模型");
        });
    },
    handleRemove(file, fileList) {
      console.log(file, fileList);
    },
    handleSuccess(file) {
      console.log(URL.createObjectURL(file.raw));
      this.filepath = URL.createObjectURL(file.raw);
    },
    handleExceed(files, fileList) {
      this.$message.warning(`只能上传一个OTU表`);
    },
    beforeRemove(file, fileList) {
      return this.$confirm(`确定移除 ${file.name}？`);
    },
  },
  watch: {
    premodel_value(val) {
      console.log("val is:", val);
      this.notprocessing = true;
      this.finished = false;
      if (val == "Nothing") {
        this.classifier_value = "";
        this.dataset_value = "";
        this.ruleForm.batchsize = "";
        this.ruleForm.epoch = "";
        this.feature = 0;
        return;
      }

      this.getData();
      let record = this.record;
      console.log("record is:", record);
      for (let i = 0; i < record.length; i++) {
        console.log("entry is:", record[i]);

        if (record[i][0] == val) {
          record = record[i];
          console.log("now record is:", record);
        }
      }

      if (record[1] != "gnn") {
        this.threshold = "";
        this.graph_classifier_value = "";
        console.log(this.graph_classifier_value + this.threshold);
      }

      if (val != "Nothing") this.dataset_value = val;
      else this.dataset_value = "";
      this.classifier_value = record[1];
      if (val != "Nothing") this.sliderValue = parseInt(record[2]);
      else this.sliderValue = 0;

      this.graph_classifier_value = record[3];
      this.ruleForm.epoch = record[4];
      this.ruleForm.batchsize = record[5];
      this.ruleForm.threshold = record[6];
    },
    classifier_value(val) {
      this.notprocessing = true;
      this.finished = false;
      if (val != "GNN") {
        this.ruleForm.epoch = "5";
        this.ruleForm.batchsize = "16";
        this.ruleForm.threshold = "_";
        this.graph_classifier_value = "_";
        console.log(val, this.graph_classifier_value + this.ruleForm.threshold);
      } else {
        this.ruleForm.epoch = "50";
        this.ruleForm.batchsize = "16";
        this.ruleForm.threshold = "40";
        this.graph_classifier_value = "DiffPool";
        console.log(this.graph_classifier_value + this.ruleForm.threshold);
      }
    },
  },
  mounted() {
    this.getData();
  },
};
</script>
<style>
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