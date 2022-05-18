import Vue from 'vue'
import Router from 'vue-router'

Vue.use(Router);

export default new Router({
  // mode: "history",
  routes: [
    {
      path: '/',
      redirect: '/index'
    },
    {
      path: '/',
      component: () => import('../components/common/Base.vue'),
      meta: {
        title: '公共部分'
      },
      children: [
        {
          path: '/index',
          component: () => import('../components/page/Home.vue'),
          meta: {
            title: '系统首页'
          }
        },
        {
          path: '/TrainHistory',
          component: () => import('../components/page/TrainHistory.vue'),
          meta: {
            title: '评估记录'
          }
        },
        {
          path: '/BestHistory',
          component: () => import('../components/page/BestHistory.vue'),
          meta: {
            title: '最优模型'
          }
        },
        {
          path: '/ModelEvaluation',
          component: () => import('../components/page/ModelEvaluation.vue'),
          meta: {
            title: '模型评估'
          }
        },
        {
          path: '/DieaseDiagnosize',
          component: () => import('../components/page/DieaseDiagnosize.vue'),
          meta: {
            title: '疾病诊断'
          }
        },
        {
          path: '/DiagnosizeHistory',
          component: () => import('../components/page/DiagnosizeHistory.vue'),
          meta: {
            title: '诊断记录'
          }
        },


        {
          path: '/SpecificDiagnosize',
          component: () => import('../components/page/SpecificDiagnosize.vue'),
          meta: {
            title: '具体结果'
          }
        },
        {
          path: '/DieaseTreatment',
          component: () => import('../components/page/DieaseTreatment.vue'),
          meta: {
            title: '诊疗建议'
          }
        },


      ]
    },
    {
      path: '/login',
      component: () => import('../components/page/Login.vue')
    },
    {
      path: '/error',
      component: () => import('../components/page/Error.vue')
    },
    {
      path: '/404',
      component: () => import('../components/page/404.vue')
    }
  ]
})
