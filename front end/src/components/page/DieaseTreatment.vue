<template>
  <div>
    <el-divider content-position="left">诊疗意见</el-divider>
    <el-collapse>
      <el-collapse-item v-for="(p, index) of Data" :title="p.name" :key="index">
        {{ p.treatment }}
      </el-collapse-item>
    </el-collapse>
  </div>
</template>
<script>
import axios from "axios";

export default {
  methods: {
    getData() {
      var that = this;
      // 对应 Python 提供的接口，这里的地址填写下面服务器运行的地址，本地则为127.0.0.1，外网则为 your_ip_address
      const path = "http://127.0.0.1:5000/DieaseTreatment";
      axios
        .post(path, {
          sampleID: that.sampleID,
          historyID: that.historyID,
        })
        .then(function (response) {
          // 这里服务器返回的 response 为一个 json object，可通过如下方法需要转成 json 字符串
          // 可以直接通过 response.data 取key-value
          // 坑一：这里不能直接使用 this 指针，不然找不到对象
          var record = response.data.record;
          console.log(record);
          // 坑二：这里直接按类型解析，若再通过 JSON.stringify(msg) 转，会得到带双引号的字串
          for (let index = 0; index < record.length; index++) {
            that.Data.splice(0, 0, {
              name: record[index][0],
              treatment: record[index][1],
            });
          }

          // alert(
          //   "Success " + response.status + ", " + response.data + ", " + record
          // );
        })
        .catch(function (error) {
          alert("Error " + error);
        });
    },
  },
  data() {
    return {
      sampleID: "",
      historyID: "",
      Data: [],
      results: "",
    };
  },
  mounted() {
    console.log(
      "in Treatment sampleID, historyID",
      this.$route.query.sampleID,
      this.$route.query.historyID
    );
    this.sampleID = this.$route.query.sampleID;
    this.historyID = this.$route.query.historyID;
    this.result = this.$route.query.result;
    this.getData();
  },
};
</script>

<style>
.el-collapse-item__header {
  background-color: #eee;
  border-bottom: 1px solid #c8cdd8;
}
.el-collapse-item__wrap {
  background-color: #eee;
  border-bottom: 1px solid #c8cdd8;
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