<template>
  <div>
    <el-table
      ref="list"
      :data="tableData"
      style="width: 100%"
      border
      stripe
      highlight-current-row
      :default-sort="{ prop: 'date', order: 'descending' }"
      @row-click="handleRowClick"
      @select-all="handleCheckedAllAndCheckedNone"
      @select="handleCheckedAllAndCheckedNone"
    >
      <el-table-column prop="ID" label="样本ID" sortable width="200">
      </el-table-column>
      <el-table-column prop="sex" label="性别"> </el-table-column>
      <el-table-column prop="age" label="年龄"> </el-table-column>
      <el-table-column prop="country" label="国家"> </el-table-column>
      <el-table-column prop="result" label="result" sortable> </el-table-column>
      <el-table-column label="详情" align="center">
        <template slot-scope="scope">
          <el-button
            type="primary"
            icon="el-icon-search"
            title="查看"
            @click="jumpTo($event, scope.row)"
            >查看</el-button
          >
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>

<style>
.el-table .warning-row {
  background: oldlace;
}

.el-table .success-row {
  background: #f0f9eb;
}
</style>

<script>
import axios from "axios";
export default {
  methods: {
    getData() {
      var that = this;
      // 对应 Python 提供的接口，这里的地址填写下面服务器运行的地址，本地则为127.0.0.1，外网则为 your_ip_address
      const path = "http://127.0.0.1:5000/SpecificDiagnosize";
      axios
        .post(path, {
          Diagnosizeid: that.DiagnosizeID,
        })
        .then(function (response) {
          // 这里服务器返回的 response 为一个 json object，可通过如下方法需要转成 json 字符串
          // 可以直接通过 response.data 取key-value
          // 坑一：这里不能直接使用 this 指针，不然找不到对象
          var record = response.data.record;
          // 坑二：这里直接按类型解析，若再通过 JSON.stringify(msg) 转，会得到带双引号的字串
          for (let index = 0; index < record.length; index++) {
            that.tableData.splice(0, 0, {
              ID: record[index][0],
              sex: record[index][1],
              age: record[index][2],
              country: record[index][3],
              result: record[index][4],
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
    jumpTo(e, row) {
      console.log(row.ID);
      if (row.result == "negative") {
        this.$message.warning(`健康人没有诊疗意见`);
        return;
      }
      this.$router.push({
        path: "/DieaseTreatment",
        query: {
          sampleID: row.ID,
          result: row.result,
          historyID: this.DiagnosizeID,
        },
      });
    },
    tableRowClassName({ row, rowIndex }) {
      if (rowIndex === 1) {
        return "warning-row";
      } else if (rowIndex === 3) {
        return "success-row";
      }
      return "";
    },
    handleRowClick(row, event, column) {
      // 仅选中当前行
      this.setCurRowChecked(row);
    },
    handleCheckedAllAndCheckedNone(selection) {
      // 当前选中仅一行时操作-（当前表格行高亮）
      1 != selection.length && this.$refs.list.setCurrentRow();
    },
    dialogClose() {
      // 清空编辑表单
      this.$refs.editForm.resetFields();
    },
    rowDel(index, row, event) {
      // 让当前删除按钮失焦
      event.target.blur();

      this.$confirm("确定要删除当前行吗？", "删除", {
        comfirmButtonText: "确定",
        cancelButtonText: "取消",
      }).then(() => {
        this.tableData.splice(row.id, 1);
        this.$message.success("删除成功");
        return false;
      });
    },
    // 选中当前行-当前行的复选框被勾选
    setCurRowChecked(row) {
      this.$refs.list.clearSelection();
      this.$refs.list.toggleRowSelection(row);
    },
  },
  data() {
    return {
      DiagnosizeID: "",
      tableData: [],
    };
  },
  mounted() {
    console.log("我挂载了且id是", this.$route.query.id);
    this.DiagnosizeID = this.$route.query.id;
    this.getData();
  },
};
</script>