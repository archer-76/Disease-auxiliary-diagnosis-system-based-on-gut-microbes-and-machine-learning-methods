<template>
  <div class="main-tags" v-if="showTag">
    <ul class="tag-list">
      <li
        class="tag-item"
        v-for="(item, index) in tagList"
        :class="{ active: isActive(item.path) }"
        :key="index"
      >
        <router-link :to="item.path" class="tag-item-title">
          {{ item.title }}
        </router-link>
        <i class="el-icon-close" @click="closeTag(index)"></i>
      </li>
    </ul>
    <el-dropdown @command="handleTag">
      <el-button size="mini" type="primary">
        标签选项<i class="el-icon-arrow-down el-icon--right"></i>
      </el-button>
      <el-dropdown-menu slot="dropdown">
        <el-dropdown-item command="closeOther">关闭其他</el-dropdown-item>
        <el-dropdown-item command="closeAll">关闭所有</el-dropdown-item>
      </el-dropdown-menu>
    </el-dropdown>
  </div>
</template>

<script>
import Bus from "../common/bus";
export default {
  name: "Tags",
  data() {
    return {
      tagList: [],
      collapse: false,
      initTag: {
        title: "系统首页",
        path: "/index",
        name: "Home",
      },
    };
  },
  methods: {
    isActive(path) {
      return path === this.$route.fullPath;
    },
    // 关闭单个标签
    closeTag(index) {
      let delItem = this.tagList.splice(index, 1)[0];
      // 删除标签后 当前选中的标签  后一个标签 前一个标签
      let curItem = this.tagList[index]
        ? this.tagList[index]
        : this.tagList[index - 1];

      if (curItem) {
        delItem.path === this.$route.fullPath &&
          this.$router.push(curItem.path);
      } else {
        // 没有前一个标签和后一个标签 关闭该标签就类似于 关闭所有
        this.closeAll();
      }
    },
    // 关闭全部标签
    closeAll() {
      this.tagList = [this.initTag];
      this.$router.push("/index");
    },
    // 关闭其他标签
    closeOther() {
      let curItem = this.tagList.filter(
        (item) => item.path === this.$route.fullPath
      );
      this.tagList = curItem;
    },
    // 设置标签
    setTag(route) {
      let isExist = this.tagList.some(
        (item) => item.path === this.$route.fullPath
      );

      !isExist &&
        this.tagList.push({
          title: route.meta.title,
          path: route.fullPath,
          name: route.matched[1].components.default.name,
        });
      Bus.$emit("tags", this.tagList);
    },
    handleTag(command) {
      "closeOther" === command ? this.closeOther() : this.closeAll();
    },
  },
  computed: {
    showTag() {
      return this.tagList.length > 0;
    },
  },
  watch: {
    $route(newValue) {
      if (
        newValue.path != "/SpecificDiagnosize" &&
        newValue.path != "/DieaseTreatment"
      ) {
        console.log("new Value's path is:", newValue.path);
        this.setTag(newValue);
      }
    },
  },
  created() {
    this.setTag(this.$route);
  },
};
</script>

<style scoped lang="less">
.main-tags {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 2;
  width: 100%;
  height: 42px;
  padding: 8px 5px;
  background-color: @boxBgColor;
  box-shadow: 1px 0 2px #ccc;
}
.tag-list {
  float: left;
  font-size: 14px !important;
}
.tag-item {
  float: left;
  padding: 3px 15px;
  border: 1px solid #ccc;
  border-radius: 5px;
  & + & {
    margin-left: 10px;
  }
  &.active {
    border-color: #333;
    background-color: #333;
    color: white !important;
  }
  &:hover {
    background-color: #444;
    color: white;
    .tag-item-title {
      color: white;
    }
  }
  .el-icon-close {
    position: relative;
    right: -5px;
    margin-left: 3px;
    cursor: pointer;
  }
}
.tag-item-title {
  color: #666;
  &.router-link-active {
    color: white;
  }
}
.el-dropdown {
  position: absolute;
  right: 5px;
}
</style>
