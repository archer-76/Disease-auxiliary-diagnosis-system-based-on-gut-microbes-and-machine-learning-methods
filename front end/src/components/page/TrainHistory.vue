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
      <el-table-column
        prop="classifier"
        label="classifier"
        width="120"
        sortable
      >
      </el-table-column>
      <el-table-column prop="epoch" label="epoch" width="80"> </el-table-column>
      <el-table-column prop="batchsize" label="batchsize" width="100">
      </el-table-column>
      <el-table-column prop="dataset" label="dataset" width="120" sortable>
      </el-table-column>
      <el-table-column prop="threshold" label="threshold" width="100">
      </el-table-column>
      <el-table-column prop="pooling" label="pooling" width="100">
      </el-table-column>
      <el-table-column prop="acc" label="acc" width="120" sortable>
      </el-table-column>
      <el-table-column prop="auc" label="auc" width="120" sortable>
      </el-table-column>
      <el-table-column prop="date" label="date" sortable>
        <template slot-scope="scope">
          <i class="el-icon-time"></i>
          <span style="margin-left: 5px">{{ scope.row.date }}</span>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="130" align="center">
        <template slot-scope="scope">
          <el-button
            circle
            icon="el-icon-delete"
            type="danger"
            title="删除"
            size="small"
            @click="rowDel(scope.$index, scope.row, $event)"
          ></el-button>
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
      this.getData();

<script>
import axios from "axios";
export default {
  data() {
    return {
      recieveData: {},
      tableData: [],
    };
  },
  methods: {
    getData() {
      var that = this;
      // 对应 Python 提供的接口，这里的地址填写下面服务器运行的地址，本地则为127.0.0.1，外网则为 your_ip_address
      const path = "http://127.0.0.1:5000/TrainHistory";
      axios
        .get(path)
        .then(function (response) {
          // 这里服务器返回的 response 为一个 json object，可通过如下方法需要转成 json 字符串
          // 可以直接通过 response.data 取key-value
          // 坑一：这里不能直接使用 this 指针，不然找不到对象
          var record = response.data.record;
          // 坑二：这里直接按类型解析，若再通过 JSON.stringify(msg) 转，会得到带双引号的字串
          for (let index = 0; index < record.length; index++) {
            that.tableData.splice(0, 0, {
              id: record[index][0],
              dataset: record[index][1],
              classifier: record[index][2],
              feature: record[index][3],
              pooling: record[index][4],
              epoch: record[index][5],
              batchsize: record[index][6],
              threshold: record[index][7],
              acc: record[index][8],
              auc: record[index][9],
              date: record[index][10],
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
  mounted() {
    console.log("mounted", this);
    this.getData();
  },
};
</script>