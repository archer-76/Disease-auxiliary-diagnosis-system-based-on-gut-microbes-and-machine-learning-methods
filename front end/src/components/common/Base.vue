<template>
  <div class="wrapper">
    <com-header></com-header>
    <com-sidebar class="sidebar"></com-sidebar>
    <div class="main" :style="{ left: collapse ? '64px' : '200px' }">
      <com-tags></com-tags>
      <keep-alive :include="tagComponent">
        <router-view class="main-cont"></router-view>
      </keep-alive>
    </div>
  </div>
</template>

<script>
import comHeader from "./Header.vue";
import comSidebar from "./Sidebar.vue";
import comTags from "./Tags.vue";
import Bus from "./bus";
export default {
  name: "Base",
  components: { comHeader, comSidebar, comTags },
  data() {
    return {
      collapse: false,
      tagComponent: [],
    };
  },
  created() {
    console.log("Base创建了");
    Bus.$on("collapse", (isCollapse) => {
      this.collapse = isCollapse;
    });
    this.$bus.$on("sendSpecificDiagnosizeID", (data) => {
      console.log("Base收到", data);
    });
    // 只有在标签页列表里的页面才使用keep-alive，关闭标签之后就不保存到内存中了
    // Bus.$on("tags", (tagList) => {
    //   for (let item of tagList) {
    //     item.name && this.tagComponent.push(item.name);
    //   }
    // });
  },
};
</script>

<style scoped lang="less">
.wrapper {
  position: relative;
  height: 100%;
  overflow: hidden;
  background-color: #eee;
}
.sidebar,
.main {
  position: absolute;
  top: 60px;
  bottom: 0;
}
.main {
  left: 200px;
  right: 0;
  padding: 20px;
  overflow: auto;
  transition: left 0.3s ease-in-out;
}
.main-cont {
  margin-top: 40px;
}
</style>
