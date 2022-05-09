'''
 @ Libs   : python3.9 -m pip install jieba -i https://mirrors.aliyun.com/pypi/simple
 @ Author : wuheping
 @ Date   : 2022/4/5
 @ Desc   : 描述
'''
from bs4 import BeautifulSoup

html = '''

    <!--主体main-->
    <div id="main" class="inner home-inner">
        <div class="home-box">
<div class="home-sider">
    <!-- 左侧职位选择 -->
    <div class="job-menu">
                        <dl class="">
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>技术</b>
                        <a href="/c101010100-p100101/">Java</a>
                        <a href="/c101010100-p100121/">语音/视频/图形开发</a>
                        <a href="/c101010100-p100124/">GIS工程师</a>
                        <a href="/c101010100-p100109/">Python</a>
                        <a href="/c101010100-p100103/">PHP</a>
                        <a href="/c101010100-p100105/">C</a>
                        <a href="/c101010100-p100114/">Node.js</a>
                        <a href="/c101010100-p100106/">C#</a>
                        <a href="/c101010100-p100108/">Hadoop</a>
                        <a href="/c101010100-p100110/">Delphi</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">技术</p>
                        <ul>
                                <li>
                                    <h4>后端开发</h4>
                                    <div class="text">
                                                <a ka="search_100101" href="/c101010100-p100101/">Java</a>
                                                <a ka="search_100102" href="/c101010100-p100102/">C++</a>
                                                <a ka="search_100103" href="/c101010100-p100103/">PHP</a>
                                                <a ka="search_100105" href="/c101010100-p100105/">C</a>
                                                <a ka="search_100106" href="/c101010100-p100106/">C#</a>
                                                <a ka="search_100107" href="/c101010100-p100107/">.NET</a>
                                                <a ka="search_100108" href="/c101010100-p100108/">Hadoop</a>
                                                <a ka="search_100109" href="/c101010100-p100109/">Python</a>
                                                <a ka="search_100110" href="/c101010100-p100110/">Delphi</a>
                                                <a ka="search_100111" href="/c101010100-p100111/">VB</a>
                                                <a ka="search_100112" href="/c101010100-p100112/">Perl</a>
                                                <a ka="search_100113" href="/c101010100-p100113/">Ruby</a>
                                                <a ka="search_100114" href="/c101010100-p100114/">Node.js</a>
                                                <a ka="search_100116" href="/c101010100-p100116/">Golang</a>
                                                <a ka="search_100119" href="/c101010100-p100119/">Erlang</a>
                                                <a ka="search_100121" href="/c101010100-p100121/">语音/视频/图形开发</a>
                                                <a ka="search_100122" href="/c101010100-p100122/">数据采集</a>
                                                <a ka="search_100123" href="/c101010100-p100123/">全栈工程师</a>
                                                <a ka="search_100124" href="/c101010100-p100124/">GIS工程师</a>
                                                <a ka="search_100199" href="/c101010100-p100199/">后端开发</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>移动开发</h4>
                                    <div class="text">
                                                <a ka="search_100201" href="/c101010100-p100201/">HTML5</a>
                                                <a ka="search_100202" href="/c101010100-p100202/">Android</a>
                                                <a ka="search_100203" href="/c101010100-p100203/">iOS</a>
                                                <a ka="search_100205" href="/c101010100-p100205/">移动web前端</a>
                                                <a ka="search_100206" href="/c101010100-p100206/">Flash开发</a>
                                                <a ka="search_100208" href="/c101010100-p100208/">JavaScript</a>
                                                <a ka="search_100209" href="/c101010100-p100209/">U3D</a>
                                                <a ka="search_100210" href="/c101010100-p100210/">Cocos</a>
                                                <a ka="search_100211" href="/c101010100-p100211/">UE4</a>
                                                <a ka="search_100299" href="/c101010100-p100299/">移动开发</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>测试</h4>
                                    <div class="text">
                                                <a ka="search_100301" href="/c101010100-p100301/">测试工程师</a>
                                                <a ka="search_100302" href="/c101010100-p100302/">自动化测试</a>
                                                <a ka="search_100303" href="/c101010100-p100303/">功能测试</a>
                                                <a ka="search_100304" href="/c101010100-p100304/">性能测试</a>
                                                <a ka="search_100305" href="/c101010100-p100305/">测试开发</a>
                                                <a ka="search_100306" href="/c101010100-p100306/">移动端测试</a>
                                                <a ka="search_100307" href="/c101010100-p100307/">游戏测试</a>
                                                <a ka="search_100308" href="/c101010100-p100308/">硬件测试</a>
                                                <a ka="search_100309" href="/c101010100-p100309/">软件测试</a>
                                                <a ka="search_100310" href="/c101010100-p100310/">渗透测试</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>运维/技术支持</h4>
                                    <div class="text">
                                                <a ka="search_100401" href="/c101010100-p100401/">运维工程师</a>
                                                <a ka="search_100402" href="/c101010100-p100402/">运维开发工程师</a>
                                                <a ka="search_100403" href="/c101010100-p100403/">网络工程师</a>
                                                <a ka="search_100404" href="/c101010100-p100404/">系统工程师</a>
                                                <a ka="search_100405" href="/c101010100-p100405/">IT技术支持</a>
                                                <a ka="search_100406" href="/c101010100-p100406/">系统管理员</a>
                                                <a ka="search_100407" href="/c101010100-p100407/">网络安全</a>
                                                <a ka="search_100408" href="/c101010100-p100408/">系统安全</a>
                                                <a ka="search_100409" href="/c101010100-p100409/">DBA</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>数据</h4>
                                    <div class="text">
                                                <a ka="search_100599" href="/c101010100-p100599/">数据</a>
                                                <a ka="search_100506" href="/c101010100-p100506/">ETL工程师</a>
                                                <a ka="search_100507" href="/c101010100-p100507/">数据仓库</a>
                                                <a ka="search_100508" href="/c101010100-p100508/">数据开发</a>
                                                <a ka="search_100509" href="/c101010100-p100509/">数据挖掘</a>
                                                <a ka="search_100511" href="/c101010100-p100511/">数据分析师</a>
                                                <a ka="search_100512" href="/c101010100-p100512/">数据架构师</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>项目管理</h4>
                                    <div class="text">
                                                <a ka="search_100601" href="/c101010100-p100601/">项目经理/主管</a>
                                                <a ka="search_100603" href="/c101010100-p100603/">项目助理</a>
                                                <a ka="search_100604" href="/c101010100-p100604/">项目专员</a>
                                                <a ka="search_100605" href="/c101010100-p100605/">实施顾问</a>
                                                <a ka="search_100606" href="/c101010100-p100606/">实施工程师</a>
                                                <a ka="search_100607" href="/c101010100-p100607/">需求分析工程师</a>
                                                <a ka="search_100817" href="/c101010100-p100817/">硬件项目经理</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>硬件开发</h4>
                                    <div class="text">
                                                <a ka="search_100801" href="/c101010100-p100801/">硬件工程师</a>
                                                <a ka="search_100802" href="/c101010100-p100802/">嵌入式</a>
                                                <a ka="search_100803" href="/c101010100-p100803/">自动化</a>
                                                <a ka="search_100804" href="/c101010100-p100804/">单片机</a>
                                                <a ka="search_100805" href="/c101010100-p100805/">电路设计</a>
                                                <a ka="search_100806" href="/c101010100-p100806/">驱动开发</a>
                                                <a ka="search_100807" href="/c101010100-p100807/">系统集成</a>
                                                <a ka="search_100808" href="/c101010100-p100808/">FPGA开发</a>
                                                <a ka="search_100809" href="/c101010100-p100809/">DSP开发</a>
                                                <a ka="search_100810" href="/c101010100-p100810/">ARM开发</a>
                                                <a ka="search_100811" href="/c101010100-p100811/">PCB工艺</a>
                                                <a ka="search_100816" href="/c101010100-p100816/">射频工程师</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>前端开发</h4>
                                    <div class="text">
                                                <a ka="search_100999" href="/c101010100-p100999/">前端开发</a>
                                                <a ka="search_100901" href="/c101010100-p100901/">web前端</a>
                                                <a ka="search_100902" href="/c101010100-p100902/">JavaScript</a>
                                                <a ka="search_100903" href="/c101010100-p100903/">Flash开发</a>
                                                <a ka="search_100904" href="/c101010100-p100904/">HTML5</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>通信</h4>
                                    <div class="text">
                                                <a ka="search_101001" href="/c101010100-p101001/">通信技术工程师</a>
                                                <a ka="search_101002" href="/c101010100-p101002/">通信研发工程师</a>
                                                <a ka="search_101003" href="/c101010100-p101003/">数据通信工程师</a>
                                                <a ka="search_101004" href="/c101010100-p101004/">移动通信工程师</a>
                                                <a ka="search_101005" href="/c101010100-p101005/">电信网络工程师</a>
                                                <a ka="search_101006" href="/c101010100-p101006/">电信交换工程师</a>
                                                <a ka="search_101007" href="/c101010100-p101007/">有线传输工程师</a>
                                                <a ka="search_101008" href="/c101010100-p101008/">无线/射频通信工程师</a>
                                                <a ka="search_101009" href="/c101010100-p101009/">通信电源工程师</a>
                                                <a ka="search_101010" href="/c101010100-p101010/">通信标准化工程师</a>
                                                <a ka="search_101011" href="/c101010100-p101011/">通信项目专员</a>
                                                <a ka="search_101012" href="/c101010100-p101012/">通信项目经理</a>
                                                <a ka="search_101013" href="/c101010100-p101013/">核心网工程师</a>
                                                <a ka="search_101014" href="/c101010100-p101014/">通信测试工程师</a>
                                                <a ka="search_101015" href="/c101010100-p101015/">通信设备工程师</a>
                                                <a ka="search_101016" href="/c101010100-p101016/">光通信工程师</a>
                                                <a ka="search_101017" href="/c101010100-p101017/">光传输工程师</a>
                                                <a ka="search_101018" href="/c101010100-p101018/">光网络工程师</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>电子/半导体</h4>
                                    <div class="text">
                                                <a ka="search_101402" href="/c101010100-p101402/">电气工程师</a>
                                                <a ka="search_101404" href="/c101010100-p101404/">电气设计工程师</a>
                                                <a ka="search_101401" href="/c101010100-p101401/">电子工程师</a>
                                                <a ka="search_101405" href="/c101010100-p101405/">集成电路IC设计</a>
                                                <a ka="search_101403" href="/c101010100-p101403/">FAE</a>
                                                <a ka="search_101406" href="/c101010100-p101406/">IC验证工程师</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>高端技术职位</h4>
                                    <div class="text">
                                                <a ka="search_100799" href="/c101010100-p100799/">高端技术职位</a>
                                                <a ka="search_100701" href="/c101010100-p100701/">技术经理</a>
                                                <a ka="search_100702" href="/c101010100-p100702/">技术总监</a>
                                                <a ka="search_100703" href="/c101010100-p100703/">测试经理</a>
                                                <a ka="search_100704" href="/c101010100-p100704/">架构师</a>
                                                <a ka="search_100705" href="/c101010100-p100705/">CTO</a>
                                                <a ka="search_100706" href="/c101010100-p100706/">运维总监</a>
                                                <a ka="search_100707" href="/c101010100-p100707/">技术合伙人</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>人工智能</h4>
                                    <div class="text">
                                                <a ka="search_101399" href="/c101010100-p101399/">人工智能</a>
                                                <a ka="search_100104" href="/c101010100-p100104/">数据挖掘</a>
                                                <a ka="search_100115" href="/c101010100-p100115/">搜索算法</a>
                                                <a ka="search_100117" href="/c101010100-p100117/">自然语言处理</a>
                                                <a ka="search_100118" href="/c101010100-p100118/">推荐算法</a>
                                                <a ka="search_100120" href="/c101010100-p100120/">算法工程师</a>
                                                <a ka="search_101308" href="/c101010100-p101308/">智能驾驶系统工程师</a>
                                                <a ka="search_101309" href="/c101010100-p101309/">反欺诈/风控算法</a>
                                                <a ka="search_101301" href="/c101010100-p101301/">机器学习</a>
                                                <a ka="search_101302" href="/c101010100-p101302/">深度学习</a>
                                                <a ka="search_101305" href="/c101010100-p101305/">语音识别</a>
                                                <a ka="search_101306" href="/c101010100-p101306/">图像识别</a>
                                                <a ka="search_101307" href="/c101010100-p101307/">算法研究员</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>销售技术支持</h4>
                                    <div class="text">
                                                <a ka="search_101299" href="/c101010100-p101299/">销售技术支持</a>
                                                <a ka="search_101201" href="/c101010100-p101201/">售前技术支持</a>
                                                <a ka="search_101202" href="/c101010100-p101202/">售后技术支持</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>其他技术职位</h4>
                                    <div class="text">
                                                <a ka="search_101101" href="/c101010100-p101101/">其他技术职位</a>
                                    </div>
                                </li>
                        </ul>
                    </div>
                </dl>
                <dl class="">
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>产品</b>
                        <a href="/c101010100-p110101/">产品经理</a>
                        <a href="/c101010100-p110302/">产品总监/VP</a>
                        <a href="/c101010100-p110105/">数据产品经理</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">产品</p>
                        <ul>
                                <li>
                                    <h4>产品经理</h4>
                                    <div class="text">
                                                <a ka="search_110101" href="/c101010100-p110101/">产品经理</a>
                                                <a ka="search_110102" href="/c101010100-p110102/">网页产品经理</a>
                                                <a ka="search_110103" href="/c101010100-p110103/">移动产品经理</a>
                                                <a ka="search_110104" href="/c101010100-p110104/">产品助理</a>
                                                <a ka="search_110105" href="/c101010100-p110105/">数据产品经理</a>
                                                <a ka="search_110106" href="/c101010100-p110106/">电商产品经理</a>
                                                <a ka="search_110107" href="/c101010100-p110107/">游戏策划</a>
                                                <a ka="search_110108" href="/c101010100-p110108/">产品专员</a>
                                                <a ka="search_110109" href="/c101010100-p110109/">硬件产品经理</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>高端产品职位</h4>
                                    <div class="text">
                                                <a ka="search_110399" href="/c101010100-p110399/">高端产品职位</a>
                                                <a ka="search_110302" href="/c101010100-p110302/">产品总监/VP</a>
                                                <a ka="search_110303" href="/c101010100-p110303/">游戏制作人</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>其他产品职位</h4>
                                    <div class="text">
                                                <a ka="search_110401" href="/c101010100-p110401/">其他产品职位</a>
                                    </div>
                                </li>
                        </ul>
                    </div>
                </dl>
                <dl class="">
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>设计</b>
                        <a href="/c101010100-p120105/">UI设计师</a>
                        <a href="/c101010100-p120106/">平面设计</a>
                        <a href="/c101010100-p120201/">交互设计师</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">设计</p>
                        <ul>
                                <li>
                                    <h4>视觉/交互设计</h4>
                                    <div class="text">
                                                <a ka="search_120199" href="/c101010100-p120199/">视觉设计</a>
                                                <a ka="search_120101" href="/c101010100-p120101/">视觉设计师</a>
                                                <a ka="search_120102" href="/c101010100-p120102/">网页设计师</a>
                                                <a ka="search_120103" href="/c101010100-p120103/">Flash设计师</a>
                                                <a ka="search_120104" href="/c101010100-p120104/">APP设计师</a>
                                                <a ka="search_120105" href="/c101010100-p120105/">UI设计师</a>
                                                <a ka="search_120106" href="/c101010100-p120106/">平面设计</a>
                                                <a ka="search_120107" href="/c101010100-p120107/">3D设计师</a>
                                                <a ka="search_120108" href="/c101010100-p120108/">广告设计</a>
                                                <a ka="search_120109" href="/c101010100-p120109/">多媒体设计师</a>
                                                <a ka="search_120110" href="/c101010100-p120110/">原画师</a>
                                                <a ka="search_120116" href="/c101010100-p120116/">CAD设计/制图</a>
                                                <a ka="search_120117" href="/c101010100-p120117/">美工</a>
                                                <a ka="search_120118" href="/c101010100-p120118/">包装设计</a>
                                                <a ka="search_120119" href="/c101010100-p120119/">设计师助理</a>
                                                <a ka="search_120120" href="/c101010100-p120120/">动画设计</a>
                                                <a ka="search_120121" href="/c101010100-p120121/">插画师</a>
                                                <a ka="search_120122" href="/c101010100-p120122/">漫画师</a>
                                                <a ka="search_120123" href="/c101010100-p120123/">人像修图师</a>
                                                <a ka="search_120201" href="/c101010100-p120201/">交互设计师</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>游戏设计</h4>
                                    <div class="text">
                                                <a ka="search_120305" href="/c101010100-p120305/">系统策划</a>
                                                <a ka="search_120111" href="/c101010100-p120111/">游戏特效</a>
                                                <a ka="search_120112" href="/c101010100-p120112/">游戏界面设计师</a>
                                                <a ka="search_120113" href="/c101010100-p120113/">游戏场景</a>
                                                <a ka="search_120114" href="/c101010100-p120114/">游戏角色</a>
                                                <a ka="search_120115" href="/c101010100-p120115/">游戏动作</a>
                                                <a ka="search_120303" href="/c101010100-p120303/">游戏数值策划</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>用户研究</h4>
                                    <div class="text">
                                                <a ka="search_120301" href="/c101010100-p120301/">数据分析师</a>
                                                <a ka="search_120302" href="/c101010100-p120302/">用户研究员</a>
                                                <a ka="search_120304" href="/c101010100-p120304/">UX设计师</a>
                                                <a ka="search_120407" href="/c101010100-p120407/">用户研究经理</a>
                                                <a ka="search_120408" href="/c101010100-p120408/">用户研究总监</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>高端设计职位</h4>
                                    <div class="text">
                                                <a ka="search_120499" href="/c101010100-p120499/">高端设计职位</a>
                                                <a ka="search_120401" href="/c101010100-p120401/">设计经理/主管</a>
                                                <a ka="search_120402" href="/c101010100-p120402/">设计总监</a>
                                                <a ka="search_120404" href="/c101010100-p120404/">视觉设计总监</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>非视觉设计</h4>
                                    <div class="text">
                                                <a ka="search_120699" href="/c101010100-p120699/">非视觉设计</a>
                                                <a ka="search_120611" href="/c101010100-p120611/">展览/展示设计</a>
                                                <a ka="search_120612" href="/c101010100-p120612/">照明设计</a>
                                                <a ka="search_120601" href="/c101010100-p120601/">服装/纺织设计</a>
                                                <a ka="search_120602" href="/c101010100-p120602/">工业设计</a>
                                                <a ka="search_120603" href="/c101010100-p120603/">橱柜设计</a>
                                                <a ka="search_120604" href="/c101010100-p120604/">家具设计</a>
                                                <a ka="search_120605" href="/c101010100-p120605/">家居设计</a>
                                                <a ka="search_120606" href="/c101010100-p120606/">珠宝设计</a>
                                                <a ka="search_120607" href="/c101010100-p120607/">室内设计</a>
                                                <a ka="search_120608" href="/c101010100-p120608/">陈列设计</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>其他设计职位</h4>
                                    <div class="text">
                                                <a ka="search_120501" href="/c101010100-p120501/">其他设计职位</a>
                                    </div>
                                </li>
                        </ul>
                    </div>
                </dl>
                <dl class="">
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>运营</b>
                        <a href="/c101010100-p130111/">新媒体运营</a>
                        <a href="/c101010100-p130102/">产品运营</a>
                        <a href="/c101010100-p130109/">网络推广</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">运营</p>
                        <ul>
                                <li>
                                    <h4>运营</h4>
                                    <div class="text">
                                                <a ka="search_130199" href="/c101010100-p130199/">运营</a>
                                                <a ka="search_130101" href="/c101010100-p130101/">用户运营</a>
                                                <a ka="search_130102" href="/c101010100-p130102/">产品运营</a>
                                                <a ka="search_130103" href="/c101010100-p130103/">数据/策略运营</a>
                                                <a ka="search_130104" href="/c101010100-p130104/">内容运营</a>
                                                <a ka="search_130105" href="/c101010100-p130105/">活动运营</a>
                                                <a ka="search_130106" href="/c101010100-p130106/">商家运营</a>
                                                <a ka="search_130107" href="/c101010100-p130107/">品类运营</a>
                                                <a ka="search_130108" href="/c101010100-p130108/">游戏运营</a>
                                                <a ka="search_130110" href="/c101010100-p130110/">网站运营</a>
                                                <a ka="search_130111" href="/c101010100-p130111/">新媒体运营</a>
                                                <a ka="search_130112" href="/c101010100-p130112/">社区运营</a>
                                                <a ka="search_130113" href="/c101010100-p130113/">微信运营</a>
                                                <a ka="search_130116" href="/c101010100-p130116/">线下拓展运营</a>
                                                <a ka="search_130117" href="/c101010100-p130117/">国内电商运营</a>
                                                <a ka="search_130118" href="/c101010100-p130118/">运营助理/专员</a>
                                                <a ka="search_130120" href="/c101010100-p130120/">内容审核</a>
                                                <a ka="search_130121" href="/c101010100-p130121/">数据标注/AI训练师</a>
                                                <a ka="search_130122" href="/c101010100-p130122/">直播运营</a>
                                                <a ka="search_130123" href="/c101010100-p130123/">车辆运营</a>
                                                <a ka="search_130124" href="/c101010100-p130124/">跨境电商运营</a>
                                                <a ka="search_170108" href="/c101010100-p170108/">视频运营</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>编辑</h4>
                                    <div class="text">
                                                <a ka="search_130299" href="/c101010100-p130299/">编辑</a>
                                                <a ka="search_130201" href="/c101010100-p130201/">主编/副主编</a>
                                                <a ka="search_130203" href="/c101010100-p130203/">文案编辑</a>
                                                <a ka="search_130204" href="/c101010100-p130204/">网站编辑</a>
                                                <a ka="search_130206" href="/c101010100-p130206/">采编</a>
                                                <a ka="search_210101" href="/c101010100-p210101/">医学编辑</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>客服</h4>
                                    <div class="text">
                                                <a ka="search_130301" href="/c101010100-p130301/">售前客服</a>
                                                <a ka="search_130302" href="/c101010100-p130302/">售后客服</a>
                                                <a ka="search_130303" href="/c101010100-p130303/">网络客服</a>
                                                <a ka="search_130304" href="/c101010100-p130304/">客服经理</a>
                                                <a ka="search_130305" href="/c101010100-p130305/">客服专员</a>
                                                <a ka="search_130306" href="/c101010100-p130306/">客服主管</a>
                                                <a ka="search_130308" href="/c101010100-p130308/">电话客服</a>
                                                <a ka="search_130309" href="/c101010100-p130309/">咨询热线/呼叫中心客服</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>高端运营职位</h4>
                                    <div class="text">
                                                <a ka="search_130499" href="/c101010100-p130499/">高端运营职位</a>
                                                <a ka="search_130402" href="/c101010100-p130402/">运营总监</a>
                                                <a ka="search_130403" href="/c101010100-p130403/">COO</a>
                                                <a ka="search_130404" href="/c101010100-p130404/">客服总监</a>
                                                <a ka="search_130405" href="/c101010100-p130405/">运营经理/主管</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>其他运营职位</h4>
                                    <div class="text">
                                                <a ka="search_130501" href="/c101010100-p130501/">其他运营职位</a>
                                    </div>
                                </li>
                        </ul>
                    </div>
                </dl>
                <dl class="">
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>市场</b>
                        <a href="/c101010100-p140101/">市场营销</a>
                        <a href="/c101010100-p140104/">市场推广</a>
                        <a href="/c101010100-p140203/">品牌公关</a>
                        <a href="/c101010100-p140604/">策划经理</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">市场</p>
                        <ul>
                                <li>
                                    <h4>政府事务</h4>
                                    <div class="text">
                                                <a ka="search_140112" href="/c101010100-p140112/">政府关系</a>
                                                <a ka="search_140801" href="/c101010100-p140801/">政策研究</a>
                                                <a ka="search_140802" href="/c101010100-p140802/">企业党建</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>市场/营销</h4>
                                    <div class="text">
                                                <a ka="search_130109" href="/c101010100-p130109/">网络推广</a>
                                                <a ka="search_140101" href="/c101010100-p140101/">市场营销</a>
                                                <a ka="search_140102" href="/c101010100-p140102/">市场策划</a>
                                                <a ka="search_140103" href="/c101010100-p140103/">市场顾问</a>
                                                <a ka="search_140104" href="/c101010100-p140104/">市场推广</a>
                                                <a ka="search_140105" href="/c101010100-p140105/">SEO</a>
                                                <a ka="search_140106" href="/c101010100-p140106/">SEM</a>
                                                <a ka="search_140107" href="/c101010100-p140107/">商务渠道</a>
                                                <a ka="search_140108" href="/c101010100-p140108/">商业数据分析</a>
                                                <a ka="search_140109" href="/c101010100-p140109/">活动策划</a>
                                                <a ka="search_140110" href="/c101010100-p140110/">网络营销</a>
                                                <a ka="search_140111" href="/c101010100-p140111/">海外市场</a>
                                                <a ka="search_140113" href="/c101010100-p140113/">APP推广</a>
                                                <a ka="search_140114" href="/c101010100-p140114/">选址开发</a>
                                                <a ka="search_140115" href="/c101010100-p140115/">游戏推广</a>
                                                <a ka="search_140315" href="/c101010100-p140315/">营销主管</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>公关媒介</h4>
                                    <div class="text">
                                                <a ka="search_140299" href="/c101010100-p140299/">公关媒介</a>
                                                <a ka="search_140201" href="/c101010100-p140201/">媒介经理</a>
                                                <a ka="search_140202" href="/c101010100-p140202/">广告客户执行</a>
                                                <a ka="search_140203" href="/c101010100-p140203/">品牌公关</a>
                                                <a ka="search_140204" href="/c101010100-p140204/">媒介专员</a>
                                                <a ka="search_140205" href="/c101010100-p140205/">活动策划执行</a>
                                                <a ka="search_140206" href="/c101010100-p140206/">媒介策划</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>会务会展</h4>
                                    <div class="text">
                                                <a ka="search_140599" href="/c101010100-p140599/">会务会展</a>
                                                <a ka="search_140502" href="/c101010100-p140502/">会议活动策划</a>
                                                <a ka="search_140503" href="/c101010100-p140503/">会议活动执行</a>
                                                <a ka="search_140505" href="/c101010100-p140505/">会展活动策划</a>
                                                <a ka="search_140506" href="/c101010100-p140506/">会展活动执行</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>广告</h4>
                                    <div class="text">
                                                <a ka="search_140699" href="/c101010100-p140699/">广告</a>
                                                <a ka="search_140612" href="/c101010100-p140612/">广告/会展项目经理</a>
                                                <a ka="search_140601" href="/c101010100-p140601/">广告创意设计</a>
                                                <a ka="search_140602" href="/c101010100-p140602/">美术指导</a>
                                                <a ka="search_140603" href="/c101010100-p140603/">广告设计</a>
                                                <a ka="search_140604" href="/c101010100-p140604/">策划经理</a>
                                                <a ka="search_140605" href="/c101010100-p140605/">广告文案</a>
                                                <a ka="search_140607" href="/c101010100-p140607/">广告制作</a>
                                                <a ka="search_140608" href="/c101010100-p140608/">媒介投放</a>
                                                <a ka="search_140609" href="/c101010100-p140609/">媒介合作</a>
                                                <a ka="search_140611" href="/c101010100-p140611/">广告审核</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>高端市场职位</h4>
                                    <div class="text">
                                                <a ka="search_140499" href="/c101010100-p140499/">高端市场职位</a>
                                                <a ka="search_140401" href="/c101010100-p140401/">市场总监</a>
                                                <a ka="search_140404" href="/c101010100-p140404/">CMO</a>
                                                <a ka="search_140405" href="/c101010100-p140405/">公关总监</a>
                                                <a ka="search_140406" href="/c101010100-p140406/">媒介总监</a>
                                                <a ka="search_140407" href="/c101010100-p140407/">创意总监</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>其他市场职位</h4>
                                    <div class="text">
                                                <a ka="search_140701" href="/c101010100-p140701/">其他市场职位</a>
                                    </div>
                                </li>
                        </ul>
                    </div>
                </dl>
                <dl class="">
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>人事/财务/行政</b>
                        <a href="/c101010100-p150104/">人力资源专员/助理</a>
                        <a href="/c101010100-p150204/">行政主管</a>
                        <a href="/c101010100-p150303/">财务顾问</a>
                        <a href="/c101010100-p150105/">培训</a>
                        <a href="/c101010100-p150107/">绩效考核</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">人事/财务/行政</p>
                        <ul>
                                <li>
                                    <h4>人力资源</h4>
                                    <div class="text">
                                                <a ka="search_150102" href="/c101010100-p150102/">招聘</a>
                                                <a ka="search_150103" href="/c101010100-p150103/">HRBP</a>
                                                <a ka="search_150104" href="/c101010100-p150104/">人力资源专员/助理</a>
                                                <a ka="search_150105" href="/c101010100-p150105/">培训</a>
                                                <a ka="search_150106" href="/c101010100-p150106/">薪资福利</a>
                                                <a ka="search_150107" href="/c101010100-p150107/">绩效考核</a>
                                                <a ka="search_150403" href="/c101010100-p150403/">人力资源经理/主管</a>
                                                <a ka="search_150406" href="/c101010100-p150406/">人力资源VP/CHO</a>
                                                <a ka="search_150108" href="/c101010100-p150108/">人力资源总监</a>
                                                <a ka="search_150109" href="/c101010100-p150109/">员工关系</a>
                                                <a ka="search_150110" href="/c101010100-p150110/">组织发展</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>行政</h4>
                                    <div class="text">
                                                <a ka="search_150201" href="/c101010100-p150201/">行政专员/助理</a>
                                                <a ka="search_150202" href="/c101010100-p150202/">前台</a>
                                                <a ka="search_150205" href="/c101010100-p150205/">经理助理</a>
                                                <a ka="search_150207" href="/c101010100-p150207/">后勤</a>
                                                <a ka="search_150401" href="/c101010100-p150401/">行政经理/主管</a>
                                                <a ka="search_150209" href="/c101010100-p150209/">行政总监</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>财务</h4>
                                    <div class="text">
                                                <a ka="search_150312" href="/c101010100-p150312/">建筑/工程会计</a>
                                                <a ka="search_150313" href="/c101010100-p150313/">税务外勤会计</a>
                                                <a ka="search_150314" href="/c101010100-p150314/">统计员</a>
                                                <a ka="search_150399" href="/c101010100-p150399/">财务</a>
                                                <a ka="search_150301" href="/c101010100-p150301/">会计</a>
                                                <a ka="search_150302" href="/c101010100-p150302/">出纳</a>
                                                <a ka="search_150303" href="/c101010100-p150303/">财务顾问</a>
                                                <a ka="search_150304" href="/c101010100-p150304/">结算会计</a>
                                                <a ka="search_150305" href="/c101010100-p150305/">税务</a>
                                                <a ka="search_150306" href="/c101010100-p150306/">审计</a>
                                                <a ka="search_150310" href="/c101010100-p150310/">成本会计</a>
                                                <a ka="search_150311" href="/c101010100-p150311/">总账会计</a>
                                                <a ka="search_150402" href="/c101010100-p150402/">财务经理/主管</a>
                                                <a ka="search_150404" href="/c101010100-p150404/">CFO</a>
                                                <a ka="search_150308" href="/c101010100-p150308/">财务总监/VP</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>法务</h4>
                                    <div class="text">
                                                <a ka="search_150203" href="/c101010100-p150203/">法务专员/助理</a>
                                                <a ka="search_150502" href="/c101010100-p150502/">律师</a>
                                                <a ka="search_150504" href="/c101010100-p150504/">法律顾问</a>
                                                <a ka="search_150506" href="/c101010100-p150506/">法务经理/主管</a>
                                                <a ka="search_150507" href="/c101010100-p150507/">法务总监</a>
                                    </div>
                                </li>
                                <li>
                                    <h4>其他职能职位</h4>
                                    <div class="text">
                                                <a ka="search_150601" href="/c101010100-p150601/">其他职能职位</a>
                                    </div>
                                </li>
                        </ul>
                    </div>
                </dl>
                <dl class="">
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>高级管理</b>
                        <a href="/c101010100-p150407/">总裁/总经理/CEO</a>
                        <a href="/c101010100-p150409/">分公司/代表处负责人</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">高级管理</p>
                        <ul>
                                <li>
                                    <h4>高级管理职位</h4>
                                    <div class="text">
                                                <a ka="search_150499" href="/c101010100-p150499/">高级管理职位</a>
                                                <a ka="search_150407" href="/c101010100-p150407/">总裁/总经理/CEO</a>
                                                <a ka="search_150408" href="/c101010100-p150408/">副总裁/副总经理/VP</a>
                                                <a ka="search_150409" href="/c101010100-p150409/">分公司/代表处负责人</a>
                                                <a ka="search_150410" href="/c101010100-p150410/">区域负责人(辖多个分公司)</a>
                                                <a ka="search_150411" href="/c101010100-p150411/">总助/CEO助理/董事长助理</a>
                                                <a ka="search_150413" href="/c101010100-p150413/">联合创始人</a>
                                                <a ka="search_150414" href="/c101010100-p150414/">董事会秘书</a>
                                    </div>
                                </li>
                        </ul>
                    </div>
                </dl>
        <div class="show-all">
            显示全部职位
        </div>
        <div class="all-box">
                <dl>
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>销售</b>
                            <a href="/c101010100-p140301/">销售专员</a>
                            <a href="/c101010100-p140302/">销售经理</a>
                            <a href="/c101010100-p140316/">销售工程师</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">销售</p>
                        <ul>
                                    <li>
                                        <h4>销售行政/商务</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p160301/">商务专员</a>
                                                    <a href="/c101010100-p160302/">商务经理</a>
                                                    <a href="/c101010100-p140309/">销售助理</a>
                                                    <a href="/c101010100-p140403/">商务总监</a>
                                                    <a href="/c101010100-p130119/">销售运营</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>房地产销售/招商</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p160401/">置业顾问</a>
                                                    <a href="/c101010100-p160403/">地产中介</a>
                                                    <a href="/c101010100-p220399/">房地产销售/招商</a>
                                                    <a href="/c101010100-p220403/">物业招商管理</a>
                                                    <a href="/c101010100-p220505/">房地产销售总监</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>服务业销售</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p160501/">服装销售</a>
                                                    <a href="/c101010100-p210406/">彩妆顾问</a>
                                                    <a href="/c101010100-p210414/">美容顾问</a>
                                                    <a href="/c101010100-p210610/">会籍顾问</a>
                                                    <a href="/c101010100-p290312/">珠宝销售</a>
                                                    <a href="/c101010100-p280103/">旅游顾问</a>
                                                    <a href="/c101010100-p210602/">瘦身顾问</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>汽车销售</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p230201/">汽车销售</a>
                                                    <a href="/c101010100-p230202/">汽车配件销售</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>广告/会展销售</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p140313/">广告销售</a>
                                                    <a href="/c101010100-p140501/">会议活动销售</a>
                                                    <a href="/c101010100-p140504/">会展活动销售</a>
                                                    <a href="/c101010100-p140610/">媒介顾问</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>金融销售</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p180801/">证券经纪人</a>
                                                    <a href="/c101010100-p180401/">信用卡销售</a>
                                                    <a href="/c101010100-p180701/">保险顾问</a>
                                                    <a href="/c101010100-p180506/">理财顾问</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>外贸销售</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p250201/">外贸经理</a>
                                                    <a href="/c101010100-p250203/">外贸业务员</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>销售</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p140399/">销售</a>
                                                    <a href="/c101010100-p140301/">销售专员</a>
                                                    <a href="/c101010100-p140303/">客户代表</a>
                                                    <a href="/c101010100-p140304/">大客户代表</a>
                                                    <a href="/c101010100-p140305/">BD经理</a>
                                                    <a href="/c101010100-p140307/">渠道销售</a>
                                                    <a href="/c101010100-p140308/">代理商销售</a>
                                                    <a href="/c101010100-p140310/">电话销售</a>
                                                    <a href="/c101010100-p140311/">销售顾问</a>
                                                    <a href="/c101010100-p140314/">网络销售</a>
                                                    <a href="/c101010100-p140316/">销售工程师</a>
                                                    <a href="/c101010100-p140317/">客户经理</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>课程销售</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p190601/">课程顾问</a>
                                                    <a href="/c101010100-p190602/">招生顾问</a>
                                                    <a href="/c101010100-p190603/">留学顾问</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>医疗销售</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p210599/">医疗销售</a>
                                                    <a href="/c101010100-p210506/">医疗器械销售</a>
                                                    <a href="/c101010100-p210502/">医药代表</a>
                                                    <a href="/c101010100-p210504/">健康顾问</a>
                                                    <a href="/c101010100-p210505/">医美咨询</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>销售管理</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p160199/">销售管理</a>
                                                    <a href="/c101010100-p140302/">销售经理</a>
                                                    <a href="/c101010100-p140402/">销售总监</a>
                                                    <a href="/c101010100-p160101/">区域总监</a>
                                                    <a href="/c101010100-p160102/">城市经理</a>
                                                    <a href="/c101010100-p160103/">销售VP</a>
                                                    <a href="/c101010100-p160104/">团队经理</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>其他销售职位</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p160201/">其他销售职位</a>
                                        </div>
                                    </li>
                        </ul>
                    </div>
                </dl>
                <dl>
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>传媒</b>
                            <a href="/c101010100-p170205/">广告文案</a>
                            <a href="/c101010100-p170201/">广告创意设计</a>
                            <a href="/c101010100-p170102/">编辑</a>
                            <a href="/c101010100-p170101/">记者/采编</a>
                            <a href="/c101010100-p170301/">媒介经理</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">传媒</p>
                        <ul>
                                    <li>
                                        <h4>采编/写作/出版</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p170199/">采编/写作/出版</a>
                                                    <a href="/c101010100-p170109/">印刷排版</a>
                                                    <a href="/c101010100-p170101/">记者/采编</a>
                                                    <a href="/c101010100-p170102/">编辑</a>
                                                    <a href="/c101010100-p170104/">作者/撰稿人</a>
                                                    <a href="/c101010100-p170105/">出版发行</a>
                                                    <a href="/c101010100-p170106/">校对录入</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>公关媒介</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p170399/">公关媒介</a>
                                                    <a href="/c101010100-p170301/">媒介经理</a>
                                                    <a href="/c101010100-p170302/">媒介专员</a>
                                                    <a href="/c101010100-p170303/">广告客户执行</a>
                                                    <a href="/c101010100-p170304/">品牌公关</a>
                                                    <a href="/c101010100-p170305/">活动策划执行</a>
                                                    <a href="/c101010100-p170306/">媒介策划</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>广告</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p170299/">广告</a>
                                                    <a href="/c101010100-p170212/">广告/会展项目经理</a>
                                                    <a href="/c101010100-p170201/">广告创意设计</a>
                                                    <a href="/c101010100-p170202/">美术指导</a>
                                                    <a href="/c101010100-p170203/">广告设计</a>
                                                    <a href="/c101010100-p170204/">策划经理</a>
                                                    <a href="/c101010100-p170205/">广告文案</a>
                                                    <a href="/c101010100-p170207/">广告制作</a>
                                                    <a href="/c101010100-p170208/">媒介投放</a>
                                                    <a href="/c101010100-p170209/">媒介合作</a>
                                                    <a href="/c101010100-p170211/">广告审核</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>影视媒体</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p170625/">带货主播</a>
                                                    <a href="/c101010100-p170699/">影视媒体</a>
                                                    <a href="/c101010100-p170617/">艺人助理</a>
                                                    <a href="/c101010100-p170620/">主持人/DJ</a>
                                                    <a href="/c101010100-p170621/">主播助理</a>
                                                    <a href="/c101010100-p170622/">灯光师</a>
                                                    <a href="/c101010100-p170623/">剪辑师</a>
                                                    <a href="/c101010100-p170624/">影视特效</a>
                                                    <a href="/c101010100-p170601/">导演/编导</a>
                                                    <a href="/c101010100-p170602/">摄影/摄像</a>
                                                    <a href="/c101010100-p170603/">视频编辑</a>
                                                    <a href="/c101010100-p170604/">音频编辑</a>
                                                    <a href="/c101010100-p170605/">经纪人</a>
                                                    <a href="/c101010100-p170606/">后期制作</a>
                                                    <a href="/c101010100-p170608/">影视发行</a>
                                                    <a href="/c101010100-p170609/">影视策划</a>
                                                    <a href="/c101010100-p170610/">主播</a>
                                                    <a href="/c101010100-p170611/">演员/配音/模特</a>
                                                    <a href="/c101010100-p170612/">化妆/造型/服装</a>
                                                    <a href="/c101010100-p170613/">放映员</a>
                                                    <a href="/c101010100-p170614/">录音/音效</a>
                                                    <a href="/c101010100-p170615/">制片人</a>
                                                    <a href="/c101010100-p170616/">编剧</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>其他传媒职位</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p170501/">其他传媒职位</a>
                                        </div>
                                    </li>
                        </ul>
                    </div>
                </dl>
                <dl>
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>金融</b>
                            <a href="/c101010100-p180101/">投资经理</a>
                            <a href="/c101010100-p180112/">投资总监</a>
                            <a href="/c101010100-p180201/">风控</a>
                            <a href="/c101010100-p180899/">证券/基金/期货</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">金融</p>
                        <ul>
                                    <li>
                                        <h4>证券/基金/期货</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p180803/">买方分析师</a>
                                                    <a href="/c101010100-p180804/">股票/期货操盘手</a>
                                                    <a href="/c101010100-p180805/">基金经理</a>
                                                    <a href="/c101010100-p180806/">投资银行业务</a>
                                                    <a href="/c101010100-p180899/">证券/基金/期货</a>
                                                    <a href="/c101010100-p180802/">卖方分析师</a>
                                                    <a href="/c101010100-p180106/">证券交易员</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>投融资</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p180199/">投融资</a>
                                                    <a href="/c101010100-p180101/">投资经理</a>
                                                    <a href="/c101010100-p180103/">行业研究</a>
                                                    <a href="/c101010100-p180104/">资产管理</a>
                                                    <a href="/c101010100-p180112/">投资总监</a>
                                                    <a href="/c101010100-p180113/">投资VP</a>
                                                    <a href="/c101010100-p180114/">投资合伙人</a>
                                                    <a href="/c101010100-p180115/">融资</a>
                                                    <a href="/c101010100-p180116/">并购</a>
                                                    <a href="/c101010100-p180117/">投后管理</a>
                                                    <a href="/c101010100-p180118/">投资助理</a>
                                                    <a href="/c101010100-p180111/">其他投融资职位</a>
                                                    <a href="/c101010100-p180119/">投资顾问</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>银行</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p180499/">银行</a>
                                                    <a href="/c101010100-p180102/">分析师</a>
                                                    <a href="/c101010100-p180402/">柜员</a>
                                                    <a href="/c101010100-p180403/">商务渠道</a>
                                                    <a href="/c101010100-p180404/">大堂经理</a>
                                                    <a href="/c101010100-p180405/">客户经理</a>
                                                    <a href="/c101010100-p180406/">信贷管理</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>保险</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p180702/">保险精算师</a>
                                                    <a href="/c101010100-p180703/">保险理赔</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>中后台</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p150307/">风控</a>
                                                    <a href="/c101010100-p180202/">法务</a>
                                                    <a href="/c101010100-p180203/">资信评估</a>
                                                    <a href="/c101010100-p180204/">合规稽查</a>
                                                    <a href="/c101010100-p180304/">清算</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>互联网金融</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p180599/">互联网金融</a>
                                                    <a href="/c101010100-p180501/">金融产品经理</a>
                                                    <a href="/c101010100-p180503/">催收员</a>
                                                    <a href="/c101010100-p180504/">分析师</a>
                                                    <a href="/c101010100-p180505/">投资经理</a>
                                                    <a href="/c101010100-p180110/">清算</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>其他金融职位</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p180601/">其他金融职位</a>
                                        </div>
                                    </li>
                        </ul>
                    </div>
                </dl>
                <dl>
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>教育培训</b>
                            <a href="/c101010100-p190101/">课程设计</a>
                            <a href="/c101010100-p190202/">教务管理</a>
                            <a href="/c101010100-p190499/">IT培训</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">教育培训</p>
                        <ul>
                                    <li>
                                        <h4>教育产品研发</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p190199/">教育产品研发</a>
                                                    <a href="/c101010100-p190101/">课程设计</a>
                                                    <a href="/c101010100-p190102/">课程编辑</a>
                                                    <a href="/c101010100-p190104/">培训研究</a>
                                                    <a href="/c101010100-p190105/">培训师</a>
                                                    <a href="/c101010100-p190107/">培训策划</a>
                                                    <a href="/c101010100-p190106/">其他教育产品研发职位</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>教育行政</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p190299/">教育行政</a>
                                                    <a href="/c101010100-p190205/">园长/副园长</a>
                                                    <a href="/c101010100-p190201/">校长/副校长</a>
                                                    <a href="/c101010100-p190202/">教务管理</a>
                                                    <a href="/c101010100-p190203/">教学管理</a>
                                                    <a href="/c101010100-p190204/">班主任/辅导员</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>教师</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p190301/">教师</a>
                                                    <a href="/c101010100-p190314/">日语教师</a>
                                                    <a href="/c101010100-p190315/">其他外语教师</a>
                                                    <a href="/c101010100-p190316/">语文教师</a>
                                                    <a href="/c101010100-p190317/">数学教师</a>
                                                    <a href="/c101010100-p190318/">物理教师</a>
                                                    <a href="/c101010100-p190319/">化学教师</a>
                                                    <a href="/c101010100-p190320/">生物教师</a>
                                                    <a href="/c101010100-p190321/">家教</a>
                                                    <a href="/c101010100-p190322/">托管老师</a>
                                                    <a href="/c101010100-p190323/">早教老师</a>
                                                    <a href="/c101010100-p190302/">助教</a>
                                                    <a href="/c101010100-p190303/">高中教师</a>
                                                    <a href="/c101010100-p190304/">初中教师</a>
                                                    <a href="/c101010100-p190305/">小学教师</a>
                                                    <a href="/c101010100-p190306/">幼教</a>
                                                    <a href="/c101010100-p190307/">理科教师</a>
                                                    <a href="/c101010100-p190308/">文科教师</a>
                                                    <a href="/c101010100-p190309/">英语教师</a>
                                                    <a href="/c101010100-p190310/">音乐教师</a>
                                                    <a href="/c101010100-p190311/">美术教师</a>
                                                    <a href="/c101010100-p190312/">体育教师</a>
                                                    <a href="/c101010100-p190313/">就业老师</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>IT培训</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p190499/">IT培训</a>
                                                    <a href="/c101010100-p190401/">JAVA培训讲师</a>
                                                    <a href="/c101010100-p190402/">Android培训讲师</a>
                                                    <a href="/c101010100-p190403/">iOS培训讲师</a>
                                                    <a href="/c101010100-p190404/">PHP培训讲师</a>
                                                    <a href="/c101010100-p190405/">.NET培训讲师</a>
                                                    <a href="/c101010100-p190406/">C++培训讲师</a>
                                                    <a href="/c101010100-p190407/">Unity 3D培训讲师</a>
                                                    <a href="/c101010100-p190408/">Web前端培训讲师</a>
                                                    <a href="/c101010100-p190409/">软件测试培训讲师</a>
                                                    <a href="/c101010100-p190410/">动漫培训讲师</a>
                                                    <a href="/c101010100-p190411/">UI设计培训讲师</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>职业培训</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p190501/">财会培训讲师</a>
                                                    <a href="/c101010100-p190502/">HR培训讲师</a>
                                                    <a href="/c101010100-p190503/">培训师</a>
                                                    <a href="/c101010100-p190504/">拓展培训</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>特长培训</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p190799/">教练</a>
                                                    <a href="/c101010100-p190708/">武术教练</a>
                                                    <a href="/c101010100-p190709/">轮滑教练</a>
                                                    <a href="/c101010100-p190710/">表演教师</a>
                                                    <a href="/c101010100-p190711/">机器人教师</a>
                                                    <a href="/c101010100-p190712/">书法教师</a>
                                                    <a href="/c101010100-p190713/">钢琴教师</a>
                                                    <a href="/c101010100-p190714/">吉他教师</a>
                                                    <a href="/c101010100-p190715/">古筝教师</a>
                                                    <a href="/c101010100-p190716/">播音主持教师</a>
                                                    <a href="/c101010100-p190717/">乐高教师</a>
                                                    <a href="/c101010100-p190701/">舞蹈老师</a>
                                                    <a href="/c101010100-p190702/">瑜伽老师</a>
                                                    <a href="/c101010100-p190704/">游泳教练</a>
                                                    <a href="/c101010100-p190705/">健身教练</a>
                                                    <a href="/c101010100-p190706/">篮球/羽毛球教练</a>
                                                    <a href="/c101010100-p190707/">跆拳道教练</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>其他教育培训职位</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p190801/">其他教育培训职位</a>
                                        </div>
                                    </li>
                        </ul>
                    </div>
                </dl>
                <dl>
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>医疗健康</b>
                            <a href="/c101010100-p210104/">药剂师</a>
                            <a href="/c101010100-p210401/">营养师/健康管理师</a>
                            <a href="/c101010100-p210105/">医疗器械研发</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">医疗健康</p>
                        <ul>
                                    <li>
                                        <h4>临床试验</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p210118/">临床研究</a>
                                                    <a href="/c101010100-p210119/">临床协调</a>
                                                    <a href="/c101010100-p210120/">临床数据分析</a>
                                                    <a href="/c101010100-p211001/">临床项目经理</a>
                                                    <a href="/c101010100-p210501/">医学总监</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>医生/医技</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p210103/">医生</a>
                                                    <a href="/c101010100-p210308/">幼儿园保健医</a>
                                                    <a href="/c101010100-p210112/">医生助理</a>
                                                    <a href="/c101010100-p210113/">放射科医生</a>
                                                    <a href="/c101010100-p210114/">超声科医生</a>
                                                    <a href="/c101010100-p210306/">内科医生</a>
                                                    <a href="/c101010100-p210307/">全科医生</a>
                                                    <a href="/c101010100-p210302/">中医</a>
                                                    <a href="/c101010100-p210303/">心理医生</a>
                                                    <a href="/c101010100-p210104/">药剂师</a>
                                                    <a href="/c101010100-p210304/">口腔科医生</a>
                                                    <a href="/c101010100-p210305/">康复治疗师</a>
                                                    <a href="/c101010100-p210109/">验光师</a>
                                                    <a href="/c101010100-p210111/">检验科医师</a>
                                                    <a href="/c101010100-p210107/">其他医生职位</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>护士/护理</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p210202/">护士长</a>
                                                    <a href="/c101010100-p210201/">护士</a>
                                                    <a href="/c101010100-p210503/">导医</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>健康整形</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p210499/">健康整形</a>
                                                    <a href="/c101010100-p210401/">营养师/健康管理师</a>
                                                    <a href="/c101010100-p210402/">整形师</a>
                                                    <a href="/c101010100-p210403/">理疗师</a>
                                                    <a href="/c101010100-p210404/">针灸推拿</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>生物制药</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p210115/">生物制药</a>
                                                    <a href="/c101010100-p210116/">药品注册</a>
                                                    <a href="/c101010100-p210117/">药品生产</a>
                                                    <a href="/c101010100-p210123/">医药项目经理</a>
                                                    <a href="/c101010100-p210108/">医药研发</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>医疗器械</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p210121/">医疗器械注册</a>
                                                    <a href="/c101010100-p210122/">医疗器械生产/质量管理</a>
                                                    <a href="/c101010100-p210105/">医疗器械研发</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>药店</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p210801/">药店店长</a>
                                                    <a href="/c101010100-p210802/">执业药师/驻店药师</a>
                                                    <a href="/c101010100-p210803/">药店店员</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>其他医疗健康职位</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p210701/">其他医疗健康职位</a>
                                        </div>
                                    </li>
                        </ul>
                    </div>
                </dl>
                <dl>
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>采购/贸易</b>
                            <a href="/c101010100-p250102/">采购经理/主管</a>
                            <a href="/c101010100-p250106/">采购主管</a>
                            <a href="/c101010100-p250299/">进出口贸易</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">采购/贸易</p>
                        <ul>
                                    <li>
                                        <h4>采购</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p250199/">采购</a>
                                                    <a href="/c101010100-p140312/">商品经理</a>
                                                    <a href="/c101010100-p250108/">供应商质量工程师</a>
                                                    <a href="/c101010100-p250101/">采购总监</a>
                                                    <a href="/c101010100-p250102/">采购经理/主管</a>
                                                    <a href="/c101010100-p250103/">采购专员/助理</a>
                                                    <a href="/c101010100-p250104/">买手</a>
                                                    <a href="/c101010100-p250105/">采购工程师</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>进出口贸易</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p250299/">进出口贸易</a>
                                                    <a href="/c101010100-p250204/">贸易跟单</a>
                                                    <a href="/c101010100-p240114/">报关/报检员</a>
                                                    <a href="/c101010100-p240117/">单证员</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>其他采购/贸易职位</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p250301/">其他采购/贸易类职位</a>
                                        </div>
                                    </li>
                        </ul>
                    </div>
                </dl>
                <dl>
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>供应链/物流</b>
                            <a href="/c101010100-p240103/">物流专员</a>
                            <a href="/c101010100-p240107/">贸易跟单</a>
                            <a href="/c101010100-p240102/">供应链经理</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">供应链/物流</p>
                        <ul>
                                    <li>
                                        <h4>物流</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p240199/">物流</a>
                                                    <a href="/c101010100-p240101/">供应链专员</a>
                                                    <a href="/c101010100-p240102/">供应链经理</a>
                                                    <a href="/c101010100-p240302/">集装箱管理</a>
                                                    <a href="/c101010100-p240103/">物流专员</a>
                                                    <a href="/c101010100-p240104/">物流经理</a>
                                                    <a href="/c101010100-p240105/">物流运营</a>
                                                    <a href="/c101010100-p240106/">物流跟单</a>
                                                    <a href="/c101010100-p240108/">调度员</a>
                                                    <a href="/c101010100-p240109/">物流/仓储项目经理</a>
                                                    <a href="/c101010100-p240111/">货运代理专员</a>
                                                    <a href="/c101010100-p240112/">货运代理经理</a>
                                                    <a href="/c101010100-p240113/">水/空/陆运操作</a>
                                                    <a href="/c101010100-p240116/">核销员</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>仓储</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p240299/">仓储</a>
                                                    <a href="/c101010100-p240201/">仓库经理</a>
                                                    <a href="/c101010100-p240204/">仓库管理员</a>
                                                    <a href="/c101010100-p240205/">仓库文员</a>
                                                    <a href="/c101010100-p240206/">配/理/拣/发货</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>交通/运输</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p240399/">交通/运输</a>
                                                    <a href="/c101010100-p150208/">商务司机</a>
                                                    <a href="/c101010100-p240305/">网约车司机</a>
                                                    <a href="/c101010100-p240306/">代驾司机</a>
                                                    <a href="/c101010100-p240307/">驾校教练</a>
                                                    <a href="/c101010100-p240301/">货运司机</a>
                                                    <a href="/c101010100-p240303/">配送员</a>
                                                    <a href="/c101010100-p240304/">快递员</a>
                                                    <a href="/c101010100-p240110/">运输经理/主管</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>高端供应链职位</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p240499/">高端供应链职位</a>
                                                    <a href="/c101010100-p240401/">供应链总监</a>
                                                    <a href="/c101010100-p240402/">物流总监</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>其他供应链职位</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p240501/">其他供应链职位</a>
                                        </div>
                                    </li>
                        </ul>
                    </div>
                </dl>
                <dl>
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>房地产/建筑</b>
                            <a href="/c101010100-p220401/">物业经理</a>
                            <a href="/c101010100-p220199/">房地产规划开发</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">房地产/建筑</p>
                        <ul>
                                    <li>
                                        <h4>房地产规划开发</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p220199/">房地产规划开发</a>
                                                    <a href="/c101010100-p220101/">房地产策划</a>
                                                    <a href="/c101010100-p220102/">地产项目管理</a>
                                                    <a href="/c101010100-p220103/">地产招投标</a>
                                                    <a href="/c101010100-p220302/">房产评估师</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>设计装修与市政建设</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p220299/">设计装修与市政建设</a>
                                                    <a href="/c101010100-p220202/">建筑工程师</a>
                                                    <a href="/c101010100-p220203/">建筑设计师</a>
                                                    <a href="/c101010100-p220204/">土木/土建/结构工程师</a>
                                                    <a href="/c101010100-p220205/">室内设计</a>
                                                    <a href="/c101010100-p220206/">园林/景观设计</a>
                                                    <a href="/c101010100-p220207/">城市规划设计</a>
                                                    <a href="/c101010100-p220208/">工程监理</a>
                                                    <a href="/c101010100-p220209/">工程造价</a>
                                                    <a href="/c101010100-p220210/">工程预算</a>
                                                    <a href="/c101010100-p220211/">资料员</a>
                                                    <a href="/c101010100-p220212/">建筑施工现场管理</a>
                                                    <a href="/c101010100-p220213/">弱电工程师</a>
                                                    <a href="/c101010100-p220214/">给排水工程师</a>
                                                    <a href="/c101010100-p220215/">暖通工程师</a>
                                                    <a href="/c101010100-p220216/">幕墙工程师</a>
                                                    <a href="/c101010100-p220217/">软装设计师</a>
                                                    <a href="/c101010100-p220218/">施工员</a>
                                                    <a href="/c101010100-p220219/">测绘/测量</a>
                                                    <a href="/c101010100-p220220/">材料员</a>
                                                    <a href="/c101010100-p220221/">BIM工程师</a>
                                                    <a href="/c101010100-p220222/">装修项目经理</a>
                                                    <a href="/c101010100-p220223/">建筑机电工程师</a>
                                                    <a href="/c101010100-p220224/">消防工程师</a>
                                                    <a href="/c101010100-p220225/">施工安全员</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>物业管理</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p220401/">物业经理</a>
                                                    <a href="/c101010100-p220404/">物业维修</a>
                                                    <a href="/c101010100-p220405/">绿化工</a>
                                                    <a href="/c101010100-p220406/">物业管理员</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>高端房地产职位</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p220599/">高端房地产职位</a>
                                                    <a href="/c101010100-p220501/">地产项目总监</a>
                                                    <a href="/c101010100-p220502/">地产策划总监</a>
                                                    <a href="/c101010100-p220503/">地产招投标总监</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>其他房地产职位</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p220601/">其他房地产职位</a>
                                        </div>
                                    </li>
                        </ul>
                    </div>
                </dl>
                <dl>
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>农/林/牧/渔</b>
                            <a href="/c101010100-p400101/">农业/林业技术员</a>
                            <a href="/c101010100-p400201/">饲养员</a>
                            <a href="/c101010100-p400203/">畜牧兽医</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">农/林/牧/渔</p>
                        <ul>
                                    <li>
                                        <h4>农业/林业</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p400101/">农业/林业技术员</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>畜牧/渔业</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p400201/">饲养员</a>
                                                    <a href="/c101010100-p400202/">禽畜/水产养殖技术员</a>
                                                    <a href="/c101010100-p400203/">畜牧兽医</a>
                                        </div>
                                    </li>
                        </ul>
                    </div>
                </dl>
                <dl>
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>咨询/翻译/法律</b>
                            <a href="/c101010100-p260101/">企业管理咨询</a>
                            <a href="/c101010100-p260201/">事务所律师</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">咨询/翻译/法律</p>
                        <ul>
                                    <li>
                                        <h4>咨询/调研</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p260199/">咨询/调研</a>
                                                    <a href="/c101010100-p260111/">知识产权/专利/商标代理人</a>
                                                    <a href="/c101010100-p260112/">心理咨询师</a>
                                                    <a href="/c101010100-p260113/">婚恋咨询师</a>
                                                    <a href="/c101010100-p260101/">企业管理咨询</a>
                                                    <a href="/c101010100-p260401/">咨询总监</a>
                                                    <a href="/c101010100-p260102/">数据分析师</a>
                                                    <a href="/c101010100-p260402/">咨询经理</a>
                                                    <a href="/c101010100-p260103/">财务咨询顾问</a>
                                                    <a href="/c101010100-p260104/">IT咨询顾问</a>
                                                    <a href="/c101010100-p260105/">人力资源咨询顾问</a>
                                                    <a href="/c101010100-p260106/">咨询项目管理</a>
                                                    <a href="/c101010100-p260107/">战略咨询</a>
                                                    <a href="/c101010100-p260108/">猎头顾问</a>
                                                    <a href="/c101010100-p260109/">市场调研</a>
                                                    <a href="/c101010100-p260110/">其他咨询顾问</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>律师</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p150503/">专利律师</a>
                                                    <a href="/c101010100-p260203/">知识产权律师</a>
                                                    <a href="/c101010100-p260204/">律师助理</a>
                                                    <a href="/c101010100-p260201/">事务所律师</a>
                                                    <a href="/c101010100-p260202/">法务</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>翻译</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p260301/">英语翻译</a>
                                                    <a href="/c101010100-p260302/">日语翻译</a>
                                                    <a href="/c101010100-p260303/">韩语/朝鲜语翻译</a>
                                                    <a href="/c101010100-p260304/">法语翻译</a>
                                                    <a href="/c101010100-p260305/">德语翻译</a>
                                                    <a href="/c101010100-p260306/">俄语翻译</a>
                                                    <a href="/c101010100-p260307/">西班牙语翻译</a>
                                                    <a href="/c101010100-p260308/">其他语种翻译</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>其他咨询类职位</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p260501/">其他咨询/翻译类职位</a>
                                        </div>
                                    </li>
                        </ul>
                    </div>
                </dl>
                <dl>
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>旅游</b>
                            <a href="/c101010100-p280103/">旅游顾问</a>
                            <a href="/c101010100-p280104/">导游</a>
                            <a href="/c101010100-p280299/">旅游产品开发/策划</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">旅游</p>
                        <ul>
                                    <li>
                                        <h4>旅游服务</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p280199/">旅游服务</a>
                                                    <a href="/c101010100-p280101/">计调</a>
                                                    <a href="/c101010100-p280102/">签证专员</a>
                                                    <a href="/c101010100-p280104/">导游</a>
                                                    <a href="/c101010100-p280105/">预定票务</a>
                                                    <a href="/c101010100-p280106/">讲解员</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>旅游产品开发/策划</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p280299/">旅游产品开发/策划</a>
                                                    <a href="/c101010100-p280201/">旅游产品经理</a>
                                                    <a href="/c101010100-p280202/">旅游策划师</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>其他旅游职位</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p280301/">其他旅游职位</a>
                                        </div>
                                    </li>
                        </ul>
                    </div>
                </dl>
                <dl>
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>服务业</b>
                            <a href="/c101010100-p290202/">服务员</a>
                            <a href="/c101010100-p290103/">客房服务员</a>
                            <a href="/c101010100-p210607/">发型师</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">服务业</p>
                        <ul>
                                    <li>
                                        <h4>餐饮</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p290223/">奶茶店店员</a>
                                                    <a href="/c101010100-p290224/">水台</a>
                                                    <a href="/c101010100-p290225/">烘焙师</a>
                                                    <a href="/c101010100-p290299/">餐饮</a>
                                                    <a href="/c101010100-p290208/">后厨</a>
                                                    <a href="/c101010100-p290209/">配菜打荷</a>
                                                    <a href="/c101010100-p290210/">茶艺师</a>
                                                    <a href="/c101010100-p290211/">西点师</a>
                                                    <a href="/c101010100-p290212/">餐饮学徒</a>
                                                    <a href="/c101010100-p290213/">面点师</a>
                                                    <a href="/c101010100-p290214/">行政总厨</a>
                                                    <a href="/c101010100-p290215/">厨师长</a>
                                                    <a href="/c101010100-p290216/">传菜员</a>
                                                    <a href="/c101010100-p290217/">洗碗工</a>
                                                    <a href="/c101010100-p290218/">凉菜厨师</a>
                                                    <a href="/c101010100-p290219/">中餐厨师</a>
                                                    <a href="/c101010100-p290220/">西餐厨师</a>
                                                    <a href="/c101010100-p290221/">日料厨师</a>
                                                    <a href="/c101010100-p290222/">烧烤师傅</a>
                                                    <a href="/c101010100-p290201/">收银</a>
                                                    <a href="/c101010100-p290202/">服务员</a>
                                                    <a href="/c101010100-p290203/">厨师</a>
                                                    <a href="/c101010100-p290204/">咖啡师</a>
                                                    <a href="/c101010100-p290205/">送餐员</a>
                                                    <a href="/c101010100-p290206/">餐饮店长</a>
                                                    <a href="/c101010100-p290207/">领班</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>酒店</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p290107/">礼仪/迎宾/接待</a>
                                                    <a href="/c101010100-p290115/">前厅经理</a>
                                                    <a href="/c101010100-p290116/">客房经理</a>
                                                    <a href="/c101010100-p290102/">酒店前台</a>
                                                    <a href="/c101010100-p290103/">客房服务员</a>
                                                    <a href="/c101010100-p290104/">酒店经理</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>零售</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p290305/">督导/巡店</a>
                                                    <a href="/c101010100-p290306/">陈列员</a>
                                                    <a href="/c101010100-p290307/">理货员</a>
                                                    <a href="/c101010100-p290308/">防损员</a>
                                                    <a href="/c101010100-p290309/">卖场经理</a>
                                                    <a href="/c101010100-p290311/">促销员</a>
                                                    <a href="/c101010100-p290302/">导购</a>
                                                    <a href="/c101010100-p290303/">店员/营业员</a>
                                                    <a href="/c101010100-p290304/">门店店长</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>美容保健</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p210407/">纹绣师</a>
                                                    <a href="/c101010100-p210408/">美体师</a>
                                                    <a href="/c101010100-p210409/">美发学徒</a>
                                                    <a href="/c101010100-p210410/">美容店长</a>
                                                    <a href="/c101010100-p210411/">足疗师</a>
                                                    <a href="/c101010100-p210412/">按摩师</a>
                                                    <a href="/c101010100-p210413/">美睫师</a>
                                                    <a href="/c101010100-p210607/">发型师</a>
                                                    <a href="/c101010100-p210608/">美甲师</a>
                                                    <a href="/c101010100-p210609/">化妆师</a>
                                                    <a href="/c101010100-p290801/">养发师</a>
                                                    <a href="/c101010100-p210405/">美容师</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>运动健身</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p210699/">健身</a>
                                                    <a href="/c101010100-p210613/">救生员</a>
                                                    <a href="/c101010100-p210601/">瑜伽老师</a>
                                                    <a href="/c101010100-p210603/">游泳教练</a>
                                                    <a href="/c101010100-p210604/">美体教练</a>
                                                    <a href="/c101010100-p210605/">舞蹈老师</a>
                                                    <a href="/c101010100-p210606/">健身教练</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>婚礼/花艺</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p290701/">花艺师</a>
                                                    <a href="/c101010100-p290702/">婚礼策划</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>宠物服务</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p290601/">宠物美容</a>
                                                    <a href="/c101010100-p290602/">宠物医生</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>安保/家政/维修</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p290105/">保安</a>
                                                    <a href="/c101010100-p290106/">保洁</a>
                                                    <a href="/c101010100-p290108/">保姆</a>
                                                    <a href="/c101010100-p290109/">月嫂</a>
                                                    <a href="/c101010100-p290110/">育婴师</a>
                                                    <a href="/c101010100-p290111/">护工</a>
                                                    <a href="/c101010100-p290112/">地铁安检</a>
                                                    <a href="/c101010100-p290113/">手机维修</a>
                                                    <a href="/c101010100-p290114/">家电维修</a>
                                                    <a href="/c101010100-p290117/">保安经理</a>
                                                    <a href="/c101010100-p290118/">产后康复师</a>
                                                    <a href="/c101010100-p290119/">钟点工</a>
                                                    <a href="/c101010100-p290120/">押运员</a>
                                                    <a href="/c101010100-p290121/">消防中控员</a>
                                                    <a href="/c101010100-p290122/">保洁经理</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>其他服务业职位</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p290313/">网吧网管</a>
                                                    <a href="/c101010100-p290401/">其他服务业职位</a>
                                        </div>
                                    </li>
                        </ul>
                    </div>
                </dl>
                <dl>
                    <dd>
                        <i class="icon-arrow-right"></i>
                        <b>生产制造</b>
                            <a href="/c101010100-p300102/">生产总监</a>
                            <a href="/c101010100-p300105/">生产员</a>
                            <a href="/c101010100-p300201/">质量管理/测试</a>
                    </dd>
                    <div class="menu-line"></div>
                    <div class="menu-sub">
                        <p class="menu-article">生产制造</p>
                        <ul>
                                    <li>
                                        <h4>生产营运</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p300199/">生产营运</a>
                                                    <a href="/c101010100-p300101/">厂长</a>
                                                    <a href="/c101010100-p300102/">生产总监</a>
                                                    <a href="/c101010100-p300103/">车间主任</a>
                                                    <a href="/c101010100-p300104/">生产组长/拉长</a>
                                                    <a href="/c101010100-p300105/">生产员</a>
                                                    <a href="/c101010100-p300106/">生产设备管理</a>
                                                    <a href="/c101010100-p300107/">生产计划管理</a>
                                                    <a href="/c101010100-p300108/">生产跟单</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>质量安全</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p300208/">质检员</a>
                                                    <a href="/c101010100-p300201/">质量管理/测试</a>
                                                    <a href="/c101010100-p300202/">可靠度工程师</a>
                                                    <a href="/c101010100-p300203/">故障分析师</a>
                                                    <a href="/c101010100-p300204/">认证工程师</a>
                                                    <a href="/c101010100-p300205/">体系工程师</a>
                                                    <a href="/c101010100-p300206/">审核员</a>
                                                    <a href="/c101010100-p300207/">生产安全员</a>
                                                    <a href="/c101010100-p230109/">汽车质量工程师</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>新能源</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p300801/">电池工程师</a>
                                                    <a href="/c101010100-p300802/">电机工程师</a>
                                                    <a href="/c101010100-p300803/">线束设计</a>
                                                    <a href="/c101010100-p300804/">充电桩设计</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>汽车制造</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p230101/">汽车设计</a>
                                                    <a href="/c101010100-p230102/">车身/造型设计</a>
                                                    <a href="/c101010100-p230103/">底盘工程师</a>
                                                    <a href="/c101010100-p230105/">动力系统工程师</a>
                                                    <a href="/c101010100-p230106/">汽车电子工程师</a>
                                                    <a href="/c101010100-p230107/">汽车零部件设计</a>
                                                    <a href="/c101010100-p230108/">汽车项目管理</a>
                                                    <a href="/c101010100-p230110/">内外饰设计工程师</a>
                                                    <a href="/c101010100-p230210/">总装工程师</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>汽车服务</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p230203/">汽车服务顾问</a>
                                                    <a href="/c101010100-p230204/">汽车维修</a>
                                                    <a href="/c101010100-p230205/">汽车美容</a>
                                                    <a href="/c101010100-p230206/">汽车定损理赔</a>
                                                    <a href="/c101010100-p230207/">二手车评估师</a>
                                                    <a href="/c101010100-p230208/">4S店店长/维修站长</a>
                                                    <a href="/c101010100-p230209/">汽车改装工程师</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>机械设计/制造</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p300399/">机械设计/制造</a>
                                                    <a href="/c101010100-p100813/">热设计工程师</a>
                                                    <a href="/c101010100-p100815/">精益工程师</a>
                                                    <a href="/c101010100-p300301/">机械工程师</a>
                                                    <a href="/c101010100-p300302/">机械设计师</a>
                                                    <a href="/c101010100-p300303/">机械设备工程师</a>
                                                    <a href="/c101010100-p300304/">机械维修/保养</a>
                                                    <a href="/c101010100-p300305/">机械制图</a>
                                                    <a href="/c101010100-p300306/">机械结构工程师</a>
                                                    <a href="/c101010100-p300307/">工业工程师</a>
                                                    <a href="/c101010100-p300308/">工艺/制程工程师</a>
                                                    <a href="/c101010100-p300309/">材料工程师</a>
                                                    <a href="/c101010100-p300310/">机电工程师</a>
                                                    <a href="/c101010100-p300311/">CNC/数控</a>
                                                    <a href="/c101010100-p300312/">冲压工程师</a>
                                                    <a href="/c101010100-p300313/">夹具工程师</a>
                                                    <a href="/c101010100-p300314/">模具工程师</a>
                                                    <a href="/c101010100-p300315/">焊接工程师</a>
                                                    <a href="/c101010100-p300316/">注塑工程师</a>
                                                    <a href="/c101010100-p300317/">铸造/锻造工程师</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>化工</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p300407/">化工项目经理</a>
                                                    <a href="/c101010100-p300401/">化工工程师</a>
                                                    <a href="/c101010100-p300402/">实验室技术员</a>
                                                    <a href="/c101010100-p300403/">化学分析</a>
                                                    <a href="/c101010100-p300404/">涂料研发</a>
                                                    <a href="/c101010100-p300405/">化妆品研发</a>
                                                    <a href="/c101010100-p300406/">食品/饮料研发</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>服装/纺织/皮革</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p300501/">服装/纺织设计</a>
                                                    <a href="/c101010100-p300507/">面料辅料开发</a>
                                                    <a href="/c101010100-p300509/">打样/制版</a>
                                                    <a href="/c101010100-p300510/">服装/纺织/皮革跟单</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>技工/普工</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p300601/">普工/操作工</a>
                                                    <a href="/c101010100-p300634/">挖掘机司机</a>
                                                    <a href="/c101010100-p300602/">叉车工</a>
                                                    <a href="/c101010100-p300603/">铲车司机</a>
                                                    <a href="/c101010100-p300604/">焊工</a>
                                                    <a href="/c101010100-p300605/">氩弧焊工</a>
                                                    <a href="/c101010100-p300606/">电工</a>
                                                    <a href="/c101010100-p300608/">木工</a>
                                                    <a href="/c101010100-p300609/">油漆工</a>
                                                    <a href="/c101010100-p300610/">车工</a>
                                                    <a href="/c101010100-p300611/">磨工</a>
                                                    <a href="/c101010100-p300612/">铣工</a>
                                                    <a href="/c101010100-p300613/">钳工</a>
                                                    <a href="/c101010100-p300614/">钻工</a>
                                                    <a href="/c101010100-p300615/">铆工</a>
                                                    <a href="/c101010100-p300616/">钣金工</a>
                                                    <a href="/c101010100-p300617/">抛光工</a>
                                                    <a href="/c101010100-p300618/">机修工</a>
                                                    <a href="/c101010100-p300619/">折弯工</a>
                                                    <a href="/c101010100-p300620/">电镀工</a>
                                                    <a href="/c101010100-p300621/">喷塑工</a>
                                                    <a href="/c101010100-p300622/">注塑工</a>
                                                    <a href="/c101010100-p300623/">组装工</a>
                                                    <a href="/c101010100-p300624/">包装工</a>
                                                    <a href="/c101010100-p300625/">空调工</a>
                                                    <a href="/c101010100-p300626/">电梯工</a>
                                                    <a href="/c101010100-p300627/">锅炉工</a>
                                                    <a href="/c101010100-p300628/">学徒工</a>
                                                    <a href="/c101010100-p300629/">缝纫工</a>
                                                    <a href="/c101010100-p300630/">搬运工</a>
                                                    <a href="/c101010100-p300631/">切割工</a>
                                                    <a href="/c101010100-p300632/">样衣工</a>
                                                    <a href="/c101010100-p300633/">模具工</a>
                                        </div>
                                    </li>
                                    <li>
                                        <h4>其他生产制造职位</h4>
                                        <div class="text">
                                                    <a href="/c101010100-p300701/">其他生产制造职位</a>
                                        </div>
                                    </li>
                        </ul>
                    </div>
                </dl>
        </div>
    </div>
</div>
<div class="home-main">
    <div class="slider-box">
        <div class="promotion-main" data-id="ea21bdcbacf046111XF90g~~">
                <table data-sort="sortDisabled"><tbody><tr class="firstRow"><td valign="top" rowspan="1" colspan="2"><a href="https://www.zhipin.com/campus/pc/?sid=web_banner_czzc03010501&shid=web_banner_czzc03010501" target="_blank" ka="banner-promotion-activity-czzc03010501"><img src="https://img.bosszhipin.com/beijin/upload/image/20220225/60a79c5ccb983c1c4c40be6d3554da15.jpg?x-oss-process=image/format,jpg"/></a></td><td width="307" valign="top" style="word-break: break-all;"><a href="https://www.zhipin.com/job_detail/?query=%E8%AE%BE%E8%AE%A1%E5%B8%88" target="_blank" ka="https://www.zhipin.com/job_detail/?query=设计师" style="display: inline-block; width: 307px; white-space: normal;"><img src="https://img.bosszhipin.com/beijin/upload/image/20191225/719b5568228bda8229408e1401457f13.jpg?x-oss-process=image/format,jpg" style="display: inline-block; width: 307px;"/></a></td></tr><tr><td valign="top" rowspan="2" colspan="2" style="word-break: break-all;"><a href="https://www.zhipin.com/job_detail/?query=%E5%A4%A7%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B%E5%B8%88" target="_blank" ka="https://www.zhipin.com/job_detail/?query=大数据工程师" style="display: inline-block; width: 634px; white-space: normal;"><img src="https://img.bosszhipin.com/beijin/upload/image/20191225/3f7fda0998317f22ec614bfc392848b9.jpg?x-oss-process=image/format,jpg" style="display: inline-block; width: 634px;"/></a></td><td width="307" valign="top" style="word-break: break-all;"><a href="https://www.zhipin.com/job_detail/?query=%E6%B8%B8%E6%88%8F%E7%AD%96%E5%88%92" target="_blank" ka="https://www.zhipin.com/job_detail/?query=游戏策划" style="display: inline-block; width: 307px; white-space: normal;"><img src="https://img.bosszhipin.com/beijin/upload/image/20191225/631e45e84ab482efaaf0a2bafb4d9219.jpg?x-oss-process=image/format,jpg" style="display: inline-block; width: 307px;"/></a></td></tr><tr><td width="307" valign="top" style="word-break: break-all;"><a href="https://www.zhipin.com/job_detail/?query=java%E5%B7%A5%E7%A8%8B%E5%B8%88" target="_blank" ka="https://www.zhipin.com/job_detail/?query=java工程师" style="display: inline-block; width: 307px; white-space: normal;"><img src="https://img.bosszhipin.com/beijin/upload/image/20191225/f1ab4555112c48503c7fdd8712531102.jpg?x-oss-process=image/format,jpg" style="display: inline-block; width: 307px;"/></a></td></tr></tbody></table><p><br/></p>
        </div>
    </div>
</div>                <div class="common-tab-box merge-city-job-recommend">
                    <div class="box-title">精选职位</div>
                    <h3>
                        <span class="cur" ka="index_rcmd_job_type_1">精选职位</span>
                        <span class="" ka="index_rcmd_job_type_2">最新职位</span>
                        <span class="" ka="index_rcmd_job_type_3">急招职位<i class="hot-icon">hot</i></span>
                        <div class="dropdown-wrap dropdown-filter-geek-recommend">
                            <div class="geek-img"><img src="https://img.bosszhipin.com/beijin/upload/avatar/20211231/607f1f3d68754fd090c6ed8f7ac32022f918841441b3dbb9e6809f289acbd838d685b01540f8b4ab_s.png" alt=""/>根据求职期望匹配：</div>
                        </div>
                    </h3>

                    <p class="common-tab-more"><a class="btn btn-outline" href="/web/geek/recommend" ka="open_joblist">查看更多</a></p>
                </div>
            <!--职位tab列表-->
            <div class="common-tab-box merge-city-job">
                <div class="box-title">热招职位</div>
                <h3><span class="cur" ka="index_rcmd_job_type_1">IT·互联网</span><span class="" ka="index_rcmd_job_type_2">金融</span><span class="" ka="index_rcmd_job_type_3">房地产·建筑</span><span class="" ka="index_rcmd_job_type_4">教育培训</span><span class="" ka="index_rcmd_job_type_5">娱乐传媒</span><span class="" ka="index_rcmd_job_type_6">医疗健康</span><span class="" ka="index_rcmd_job_type_7">法律咨询</span><span class="" ka="index_rcmd_job_type_8">供应链·物流</span><span class="" ka="index_rcmd_job_type_9">采购贸易</span></h3>
                        <ul class="cur">
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/54f21ad1f79b9cc41XV439S0EVVR.html" ka="index_rcmd_job_1" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">嵌入式软件工程师</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">12-24K·13薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>1-3年<span class="vline"></span>硕士</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/26d2f82213b910770HF63d24.html" ka="index_rcmd_company_1" class="user-info" target="_blank">
                                                <p><img src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20170622/b0b8220557a4885be7c5a47dc1be5ed78557dfe1cc952d785f0cb82a7a6a5197.jpg" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20170622/b0b8220557a4885be7c5a47dc1be5ed78557dfe1cc952d785f0cb82a7a6a5197.jpg?x-oss-process=image/resize,w_60,limit_0" alt="联发科技"></p>
                                            </a>
                                            <a href="/gongsi/26d2f82213b910770HF63d24.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">联发科技</span>
                                                <span class="type">其他行业</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/051c769f46ed41320HVy09m4GVM~.html" ka="index_rcmd_job_2" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">Java高级开发工程师</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-40K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/c73a957986af34b31XN53Nq1EA~~.html" ka="index_rcmd_company_2" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20201202/df98455264b9282eaec79ddbafb2514c829336cb37f50b4a92d832ac281a2b70_s.jpg?x-oss-process=image/resize,w_60,limit_0" alt="光大科技"></p>
                                            </a>
                                            <a href="/gongsi/c73a957986af34b31XN53Nq1EA~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">光大科技</span>
                                                <span class="type">互联网金融</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/13bd270c3de436000X1z0t21EFc~.html" ka="index_rcmd_job_3" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">产品经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-30K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>经验不限<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/053ea84b6939834633x83tk~.html" ka="index_rcmd_company_3" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20190215/52960859e7f5da287634478ae4177342cfcd208495d565ef66e7dff9f98764da.jpg?x-oss-process=image/resize,w_60,limit_0" alt="阿里巴巴高德"></p>
                                            </a>
                                            <a href="/gongsi/053ea84b6939834633x83tk~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">阿里巴巴高德</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/4fff6166f44d72211XVz09m5EFVQ.html" ka="index_rcmd_job_4" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">海外运营（社群/新媒体运营）</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-30K·14薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/5acba253df2515a61HV_2t--.html" ka="index_rcmd_company_4" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/bar/20200916/9419b0e24cc0b4324a3823d07fd44b4abe1bd4a3bd2a63f070bdbdada9aad826.png?x-oss-process=image/resize,w_60,limit_0" alt="火花思维"></p>
                                            </a>
                                            <a href="/gongsi/5acba253df2515a61HV_2t--.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">火花思维</span>
                                                <span class="type">在线教育</span>
                                                <span class="vline"></span>
                                                <span class="level">D轮及以上</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/117dc8d82baf52e61XR73tS5GFZR.html" ka="index_rcmd_job_5" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">网络工程师（北京）</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">13-26K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/564018fcde2fcd9f1nZ83N61GFU~.html" ka="index_rcmd_company_5" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/upload/com/logo/20201127/b88d3765fd8c094b6004f0b22ef2536b40f90dcb91a28ea8f203eb23e43cb1b8.png?x-oss-process=image/resize,w_60,limit_0" alt="软通动力信息技术集团"></p>
                                            </a>
                                            <a href="/gongsi/564018fcde2fcd9f1nZ83N61GFU~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">软通动力信息技术集团</span>
                                                <span class="type">计算机软件</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/99ee103278cf9b411XRz3NS-GVJR.html" ka="index_rcmd_job_6" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">数据安全分析师</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">25-45K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>1-3年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/fa2f92669c66eee31Hc~.html" ka="index_rcmd_company_6" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/banner/3e9d37e9effaa2b6daf43f3f03f7cb15cfcd208495d565ef66e7dff9f98764da.jpg?x-oss-process=image/resize,w_60,limit_0" alt="BOSS直聘"></p>
                                            </a>
                                            <a href="/gongsi/fa2f92669c66eee31Hc~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">BOSS直聘</span>
                                                <span class="type">人力资源服务</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/2ed88c76c1ad468e1nZ73t28GFJV.html" ka="index_rcmd_job_7" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">智能办公平台部_C++后端研发工程师</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">25-50K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/ab9fdc6f043679990HY~.html" ka="index_rcmd_company_7" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/banner/00c9c1238ae2c986f3e7741be97a9669cfcd208495d565ef66e7dff9f98764da.jpg?x-oss-process=image/resize,w_60,limit_0" alt="百度"></p>
                                            </a>
                                            <a href="/gongsi/ab9fdc6f043679990HY~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">百度</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/f34eeaafd0840d6a1XV-3Ny7FlpV.html" ka="index_rcmd_job_8" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">高级直播运营经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">11-22K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/38bf2d9641330eef1XZ-2A~~.html" ka="index_rcmd_company_8" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/bar/20190702/6261090e70ee610815076fbaddae1a74be1bd4a3bd2a63f070bdbdada9aad826.jpg?x-oss-process=image/resize,w_60,limit_0" alt="叮当快药"></p>
                                            </a>
                                            <a href="/gongsi/38bf2d9641330eef1XZ-2A~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">叮当快药</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">C轮</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/9a60566c838aaab91Xx42dq5FlQ~.html" ka="index_rcmd_job_9" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">前端高级工程师/专家-应用生态</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-40K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/01da85cd2b2d314a1HN40w~~.html" ka="index_rcmd_company_9" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/logo/35de95244eb9fc821dcd844c1b3acd64be1bd4a3bd2a63f070bdbdada9aad826.jpg?x-oss-process=image/resize,w_60,limit_0" alt="高德地图"></p>
                                            </a>
                                            <a href="/gongsi/01da85cd2b2d314a1HN40w~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">高德地图</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                        </ul>
                        <ul class="">
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/160aa4fc16b4e6201XZ629i0FVtW.html" ka="index_rcmd_job_10" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">行业分析师</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-25K·16薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>硕士</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/49f5302022cbbcd31HVy0966.html" ka="index_rcmd_company_10" class="user-info" target="_blank">
                                                <p><img src="https://img.bosszhipin.com/beijin/upload/com/workfeel/20210907/7bf6f160950405e9ca8283c9ff631404eefc9d72268a32f43acb070a664e880e7507fb3687058fcc.jpg" data-src="https://img.bosszhipin.com/beijin/upload/com/workfeel/20210907/7bf6f160950405e9ca8283c9ff631404eefc9d72268a32f43acb070a664e880e7507fb3687058fcc.jpg?x-oss-process=image/resize,w_60,limit_0" alt="中指研究院"></p>
                                            </a>
                                            <a href="/gongsi/49f5302022cbbcd31HVy0966.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">中指研究院</span>
                                                <span class="type">企业服务</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/c96f369b050053cf1nF53N6_GFNW.html" ka="index_rcmd_job_11" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">风控算法工程师</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-40K·15薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/3e0db371221babc603Z_3tu4.html" ka="index_rcmd_company_11" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20200103/b5b3697aca8dafe18e6d2d5057a455ae3b846c3945893110cf5040184029fd2a_s.jpg?x-oss-process=image/resize,w_60,limit_0" alt="百望股份"></p>
                                            </a>
                                            <a href="/gongsi/3e0db371221babc603Z_3tu4.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">百望股份</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">C轮</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/b8ba7a747566c3b90XJ-2NS8Eg~~.html" ka="index_rcmd_job_12" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">高级理财经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-25K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>经验不限<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/374c58e8f58a25201nN-3d-7.html" ka="index_rcmd_company_12" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20160614/abec9aca10d5e2fbe94b8a0e88e85ec6545d5c6206ca594291bdc6f1d7a24157.jpg?x-oss-process=image/resize,w_60,limit_0" alt="星展银行"></p>
                                            </a>
                                            <a href="/gongsi/374c58e8f58a25201nN-3d-7.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">星展银行</span>
                                                <span class="type">银行</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/3622173bcdee59fc1nN63du5EFNW.html" ka="index_rcmd_job_13" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">风控产品经理（银行）</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">18-30K·15薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/128a8545441481da1XR73N65FVc~.html" ka="index_rcmd_company_13" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/upload/com/workfeel/20220222/7bf6f160950405e99e994a57117d3d7a86adad0f4a267420a394344e92eb52b7064a5d17fcbcbe7b.jpg?x-oss-process=image/resize,w_60,limit_0" alt="邦盛科技"></p>
                                            </a>
                                            <a href="/gongsi/128a8545441481da1XR73N65FVc~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">邦盛科技</span>
                                                <span class="type">计算机服务</span>
                                                <span class="vline"></span>
                                                <span class="level">C轮</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/e348104a3e8b015f1XV52Nq_E1JS.html" ka="index_rcmd_job_14" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">风控策略</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-30K·15薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/368e4914783fa25133J509s~.html" ka="index_rcmd_company_14" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/banner/cff25504c2338966315c58bfb5520472cfcd208495d565ef66e7dff9f98764da.png?x-oss-process=image/resize,w_60,limit_0" alt="唱吧"></p>
                                            </a>
                                            <a href="/gongsi/368e4914783fa25133J509s~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">唱吧</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">D轮及以上</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/8066b25dd2bece551XVz3Nm_E1RV.html" ka="index_rcmd_job_15" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">风控经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-25K·13薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/34bf4e0b98d122ec1nd72di8FQ~~.html" ka="index_rcmd_company_15" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20200221/19273dc489a32990ee6ba9bf9197b18ca1210c6365d13893db5e808ce3e3ea9a_s.jpg?x-oss-process=image/resize,w_60,limit_0" alt="天九共享控股集团"></p>
                                            </a>
                                            <a href="/gongsi/34bf4e0b98d122ec1nd72di8FQ~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">天九共享控股集团</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/d1a4481c11663f5b1n182t26GVFV.html" ka="index_rcmd_job_16" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">金融量化咨询</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">12-24K·13薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>1-3年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/65c1ec8cab2cb5ac1HZ62tS-.html" ka="index_rcmd_company_16" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20160714/1a299758d87bba0d6b15755dd7a7fa97fdccc6a0374a133e26f9222c600c34ba.jpg?x-oss-process=image/resize,w_60,limit_0" alt="安永"></p>
                                            </a>
                                            <a href="/gongsi/65c1ec8cab2cb5ac1HZ62tS-.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">安永</span>
                                                <span class="type">企业服务</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/9eadcde5579db25c1nN40t27EFNW.html" ka="index_rcmd_job_17" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">直播讲师</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">12-24K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>1-3年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/da9e80d7761c27701Hx-3w~~.html" ka="index_rcmd_company_17" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/logo/e584c589eb0abb719f1b319eb52f1da6be1bd4a3bd2a63f070bdbdada9aad826.jpg?x-oss-process=image/resize,w_60,limit_0" alt="尚德机构"></p>
                                            </a>
                                            <a href="/gongsi/da9e80d7761c27701Hx-3w~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">尚德机构</span>
                                                <span class="type">在线教育</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/0e9e4435a515f99a1XR52du1E1BQ.html" ka="index_rcmd_job_18" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">金融合规产品经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-40K·14薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/3d7f204d36f8c0141nB-39q8.html" ka="index_rcmd_company_18" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/upload/com/workfeel/20211201/7bf6f160950405e95c8bd864f6c31c6ff21b87dfb5c127f118c5e5047634be6b13878b454ca613be.jpg?x-oss-process=image/resize,w_60,limit_0" alt="老虎国际"></p>
                                            </a>
                                            <a href="/gongsi/3d7f204d36f8c0141nB-39q8.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">老虎国际</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                        </ul>
                        <ul class="">
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/0d63bbcfc772d10e1XV83N2_EVNX.html" ka="index_rcmd_job_19" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">区域物业高级经理/副总监</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-25K·13薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>大专</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/c11f077d3cc00f0d0Hd_2d2_EA~~.html" ka="index_rcmd_company_19" class="user-info" target="_blank">
                                                <p><img src="https://img.bosszhipin.com/beijin/upload/com/logo/20200303/4f41b4ee033d35bff51efabfe6313ce019a1e285985be7eb06961ba0e4b43186.jpg" data-src="https://img.bosszhipin.com/beijin/upload/com/logo/20200303/4f41b4ee033d35bff51efabfe6313ce019a1e285985be7eb06961ba0e4b43186.jpg?x-oss-process=image/resize,w_60,limit_0" alt="锦和商业"></p>
                                            </a>
                                            <a href="/gongsi/c11f077d3cc00f0d0Hd_2d2_EA~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">锦和商业</span>
                                                <span class="type">房地产开发</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/93755b8deb0ad82e1nx72d6_EFZV.html" ka="index_rcmd_job_20" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">合动力2.0专项计划</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">30-40K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/0ae581d3b0d626e91Xxy39y_.html" ka="index_rcmd_company_20" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/bar/20170715/d847c3f61e9912ec20f62e7ac9ab41c8f04b3936faf229b40f090e5e43f36f3a.jpg?x-oss-process=image/resize,w_60,limit_0" alt="合生创展"></p>
                                            </a>
                                            <a href="/gongsi/0ae581d3b0d626e91Xxy39y_.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">合生创展</span>
                                                <span class="type">工程施工</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/d902f3c5051032df1XV83t-5GFdW.html" ka="index_rcmd_job_21" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">宅修疏通工程师/学徒</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">11-22K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>经验不限<span class="vline"></span>学历不限</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/0ee0d02bf18258ec1nZ_3tS9Ew~~.html" ka="index_rcmd_company_21" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/upload/com/workfeel/20220315/7bf6f160950405e9c3c53b5b4ebad52f07b0222274be9256d2bfcd1f160aba3f7f8af4bf46146576.jpg?x-oss-process=image/resize,w_60,limit_0" alt="啄木鸟家庭维修"></p>
                                            </a>
                                            <a href="/gongsi/0ee0d02bf18258ec1nZ_3tS9Ew~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">啄木鸟家庭维修</span>
                                                <span class="type">生活服务</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/93714ead69ebb3731XRy09S4F1dQ.html" ka="index_rcmd_job_22" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">建筑设计师，方案主创，施工图项目负责人</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">11-22K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>经验不限<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/818eedf5ca39a9aa1HJ53dW-Fg~~.html" ka="index_rcmd_company_22" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20191011/60291d0bd507c47af7714e1ebf1912f270c296da19b945c120664ee285575753_s.jpg?x-oss-process=image/resize,w_60,limit_0" alt="维拓设计"></p>
                                            </a>
                                            <a href="/gongsi/818eedf5ca39a9aa1HJ53dW-Fg~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">维拓设计</span>
                                                <span class="type">建筑设计</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/ae6eba632470bdef1XVy2Nm5ElVV.html" ka="index_rcmd_job_23" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">软装设计师</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-25K·15薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/7afa6c6c00c57f351XR83tS9GVA~.html" ka="index_rcmd_company_23" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/banner/bd92df1908df5ec97ea6347aa7180732cfcd208495d565ef66e7dff9f98764da.png?x-oss-process=image/resize,w_60,limit_0" alt="自如网"></p>
                                            </a>
                                            <a href="/gongsi/7afa6c6c00c57f351XR83tS9GVA~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">自如网</span>
                                                <span class="type">O2O</span>
                                                <span class="vline"></span>
                                                <span class="level">D轮及以上</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/4ace97ca986b60061nJ_2N--ElNQ.html" ka="index_rcmd_job_24" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">景观设计主管</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">16-25K·13薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/a78bfdf1952a421c1Xdz09i0.html" ka="index_rcmd_company_24" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/banner/76cf7eec086af3850e46af3c870f4e49cfcd208495d565ef66e7dff9f98764da.jpg?x-oss-process=image/resize,w_60,limit_0" alt="龙湖集团"></p>
                                            </a>
                                            <a href="/gongsi/a78bfdf1952a421c1Xdz09i0.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">龙湖集团</span>
                                                <span class="type">工程施工</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/39851be248793b941nF-2d-4FVJQ.html" ka="index_rcmd_job_25" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">业务经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-25K·13薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/2486b68c0f80f7521n1-09i0.html" ka="index_rcmd_company_25" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20200323/5f43fb4b1eb17c2702950a7ab9aea4ba5568e51dffd748405d137d0e01d95e40_s.jpg?x-oss-process=image/resize,w_60,limit_0" alt="博锐尚格"></p>
                                            </a>
                                            <a href="/gongsi/2486b68c0f80f7521n1-09i0.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">博锐尚格</span>
                                                <span class="type">计算机软件</span>
                                                <span class="vline"></span>
                                                <span class="level">D轮及以上</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/3c64ae36f46ff6c91n193Nm6EFdV.html" ka="index_rcmd_job_26" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">高端住宅物业项目经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-25K·13薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>学历不限</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/46bef247c4c713fa1XB4298~.html" ka="index_rcmd_company_26" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/upload/com/logo/20200703/8b698acc7261cbf1b4fba14af03f1f879864150573daf1a86a6bac713c29338a.png?x-oss-process=image/resize,w_60,limit_0" alt="第一物业"></p>
                                            </a>
                                            <a href="/gongsi/46bef247c4c713fa1XB4298~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">第一物业</span>
                                                <span class="type">物业服务</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/350917a259edadff1XV83t--EFNW.html" ka="index_rcmd_job_27" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">门窗维修工程师/学徒</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">11-22K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>经验不限<span class="vline"></span>学历不限</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/0ee0d02bf18258ec1nZ_3tS9Ew~~.html" ka="index_rcmd_company_27" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/upload/com/workfeel/20220315/7bf6f160950405e9c3c53b5b4ebad52f07b0222274be9256d2bfcd1f160aba3f7f8af4bf46146576.jpg?x-oss-process=image/resize,w_60,limit_0" alt="啄木鸟家庭维修"></p>
                                            </a>
                                            <a href="/gongsi/0ee0d02bf18258ec1nZ_3tS9Ew~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">啄木鸟家庭维修</span>
                                                <span class="type">生活服务</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                        </ul>
                        <ul class="">
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/1c71e3c428788bc21XV40ty7ElFR.html" ka="index_rcmd_job_28" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">教研经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-40K·16薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/ccb3335c82c3f7ac1XV83du4.html" ka="index_rcmd_company_28" class="user-info" target="_blank">
                                                <p><img src="https://img.bosszhipin.com/beijin/mcs/bar/20161215/4b47d0f4332635ed74dcb34f94352296b86c351e879598a87f85894b3e6f7279.jpg" data-src="https://img.bosszhipin.com/beijin/mcs/bar/20161215/4b47d0f4332635ed74dcb34f94352296b86c351e879598a87f85894b3e6f7279.jpg?x-oss-process=image/resize,w_60,limit_0" alt="百度在线"></p>
                                            </a>
                                            <a href="/gongsi/ccb3335c82c3f7ac1XV83du4.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">百度在线</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/7853836ef04b71721XV43N64EFZR.html" ka="index_rcmd_job_29" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">业务培训专家</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-35K·15薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/b633a34f787d94f21nZ_0929E1I~.html" ka="index_rcmd_company_29" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/upload/com/logo/20210420/f9bff82be5bd8c7fd799be43dae6079577c47d994acd32f441228b04529816d2.png?x-oss-process=image/resize,w_60,limit_0" alt="美团"></p>
                                            </a>
                                            <a href="/gongsi/b633a34f787d94f21nZ_0929E1I~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">美团</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/cee00b26134590691XV72tW6E1JX.html" ka="index_rcmd_job_30" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">资深课程编辑-TOB(JSQ2V)</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-26K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/c2fcd9611fb7116003J92N26FQ~~.html" ka="index_rcmd_company_30" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/banner/89d7c08c60d922a3bb281fbb45b08197cfcd208495d565ef66e7dff9f98764da.jpg?x-oss-process=image/resize,w_60,limit_0" alt="得到"></p>
                                            </a>
                                            <a href="/gongsi/c2fcd9611fb7116003J92N26FQ~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">得到</span>
                                                <span class="type">移动互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">D轮及以上</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/b279a65880fc7d2c1XVz09W1FlNR.html" ka="index_rcmd_job_31" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">升学规划讲师</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-40K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>经验不限<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/38bd5c757efa4ab6331z.html" ka="index_rcmd_company_31" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/banner/4201efdb842badc0697ac5f6db469402cfcd208495d565ef66e7dff9f98764da.jpg?x-oss-process=image/resize,w_60,limit_0" alt="网易"></p>
                                            </a>
                                            <a href="/gongsi/38bd5c757efa4ab6331z.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">网易</span>
                                                <span class="type">移动互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/5dc2a3c3986590d31n1y2dm9F1ZQ.html" ka="index_rcmd_job_32" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">健身教练</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">12-23K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>经验不限<span class="vline"></span>大专</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/41824ab6fa62132d1nN529-0ElA~.html" ka="index_rcmd_company_32" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/banner/5ff8744d97e219265a99da3f9d47d2dbcfcd208495d565ef66e7dff9f98764da.jpg?x-oss-process=image/resize,w_60,limit_0" alt="乐刻运动健身"></p>
                                            </a>
                                            <a href="/gongsi/41824ab6fa62132d1nN529-0ElA~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">乐刻运动健身</span>
                                                <span class="type">O2O</span>
                                                <span class="vline"></span>
                                                <span class="level">D轮及以上</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/3fe22722de4acd731XV_3Nq_F1pV.html" ka="index_rcmd_job_33" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">95后🌹高薪 球类爱好者（提供住宿）</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-30K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>经验不限<span class="vline"></span>大专</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/546e86b7845adde13nN439m4.html" ka="index_rcmd_company_33" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/bar/20200618/1f4c626db5058ce7e60102ac5ac88b6ebe1bd4a3bd2a63f070bdbdada9aad826.png?x-oss-process=image/resize,w_60,limit_0" alt="花香盛世体育"></p>
                                            </a>
                                            <a href="/gongsi/546e86b7845adde13nN439m4.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">花香盛世体育</span>
                                                <span class="type">在线教育</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/47b08bac40893f261nF539y0FlBW.html" ka="index_rcmd_job_34" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">小学数学主讲</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-30K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>经验不限<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/f3978df4898da0993nF_2g~~.html" ka="index_rcmd_company_34" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20190718/b3b2abf95863d99403f24c45b7395a5ffc389ab80b4049a90d7da9be2a3cbf9d_s.jpg?x-oss-process=image/resize,w_60,limit_0" alt="小盒科技"></p>
                                            </a>
                                            <a href="/gongsi/f3978df4898da0993nF_2g~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">小盒科技</span>
                                                <span class="type">移动互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">D轮及以上</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/bf4b21c89f93fe5f1nJ82N66FVRR.html" ka="index_rcmd_job_35" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">企业大学校长</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">25-40K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/6d8f3105b3b7686e33x639S_.html" ka="index_rcmd_company_35" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20190823/7a2ea657cbdcd5f9f70cad3b108d0e9ac6a167bc2bdca59546cd3b2b70de4ff3_s.jpg?x-oss-process=image/resize,w_60,limit_0" alt="星晨智慧物联网"></p>
                                            </a>
                                            <a href="/gongsi/6d8f3105b3b7686e33x639S_.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">星晨智慧物联网</span>
                                                <span class="type">医疗健康</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/11e243c0438d7be71nJ409q_EVNW.html" ka="index_rcmd_job_36" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">项目总监</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">25-50K·13薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/69b6a07fcb083c761XF809u0.html" ka="index_rcmd_company_36" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/bar/20200916/fd75c0e9015011bf30b3f75e1b13ad27be1bd4a3bd2a63f070bdbdada9aad826.png?x-oss-process=image/resize,w_60,limit_0" alt="环球网校"></p>
                                            </a>
                                            <a href="/gongsi/69b6a07fcb083c761XF809u0.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">环球网校</span>
                                                <span class="type">在线教育</span>
                                                <span class="vline"></span>
                                                <span class="level">D轮及以上</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                        </ul>
                        <ul class="">
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/474d771895cb4d5b1nFy3Nu1E1BV.html" ka="index_rcmd_job_37" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">招聘行业市场研究员</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">25-50K·14薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/fa2f92669c66eee31Hc~.html" ka="index_rcmd_company_37" class="user-info" target="_blank">
                                                <p><img src="https://img.bosszhipin.com/beijin/mcs/banner/3e9d37e9effaa2b6daf43f3f03f7cb15cfcd208495d565ef66e7dff9f98764da.jpg" data-src="https://img.bosszhipin.com/beijin/mcs/banner/3e9d37e9effaa2b6daf43f3f03f7cb15cfcd208495d565ef66e7dff9f98764da.jpg?x-oss-process=image/resize,w_60,limit_0" alt="BOSS直聘"></p>
                                            </a>
                                            <a href="/gongsi/fa2f92669c66eee31Hc~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">BOSS直聘</span>
                                                <span class="type">人力资源服务</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/a11b73a0a00ff5ee1n180tS5EFJQ.html" ka="index_rcmd_job_38" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">主持人</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-25K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/34bf4e0b98d122ec1nd72di8FQ~~.html" ka="index_rcmd_company_38" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20200221/19273dc489a32990ee6ba9bf9197b18ca1210c6365d13893db5e808ce3e3ea9a_s.jpg?x-oss-process=image/resize,w_60,limit_0" alt="天九共享控股集团"></p>
                                            </a>
                                            <a href="/gongsi/34bf4e0b98d122ec1nd72di8FQ~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">天九共享控股集团</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/8f29e99abaae9f7a1XR_2NW-FVBR.html" ka="index_rcmd_job_39" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">演艺编导</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">18-30K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/6e888338434f2e891nR52t-0.html" ka="index_rcmd_company_39" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/bar/20200916/7ad8fb2f46da3bd1f3f367b5b7b5d564be1bd4a3bd2a63f070bdbdada9aad826.png?x-oss-process=image/resize,w_60,limit_0" alt="泡泡玛特POP MART"></p>
                                            </a>
                                            <a href="/gongsi/6e888338434f2e891nR52t-0.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">泡泡玛特POP MART</span>
                                                <span class="type">新零售</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/f12c2b74091865801XV52d28ElJR.html" ka="index_rcmd_job_40" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">音频经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">25-35K·13薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>大专</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/0fe0be59e2d580de1nV72tu9.html" ka="index_rcmd_company_40" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/bar/20180719/3ea62de4a2d1a8c968f14ad05b401400be1bd4a3bd2a63f070bdbdada9aad826.png?x-oss-process=image/resize,w_60,limit_0" alt="龙图游戏"></p>
                                            </a>
                                            <a href="/gongsi/0fe0be59e2d580de1nV72tu9.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">龙图游戏</span>
                                                <span class="type">游戏</span>
                                                <span class="vline"></span>
                                                <span class="level">D轮及以上</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/96801c15d24c75301nN_2ty6EVNX.html" ka="index_rcmd_job_41" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">市场品牌经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-40K·13薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/97bce10e039a3a401HZ62t0~.html" ka="index_rcmd_company_41" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/banner/7fbf68cdb1385aa8f3234d93386d5d8bcfcd208495d565ef66e7dff9f98764da.jpg?x-oss-process=image/resize,w_60,limit_0" alt="途游游戏"></p>
                                            </a>
                                            <a href="/gongsi/97bce10e039a3a401HZ62t0~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">途游游戏</span>
                                                <span class="type">游戏</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/1c41114f56381c703nV909W6GFQ~.html" ka="index_rcmd_job_42" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">高级编辑（汽车频道）</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-24K·14薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/1009ea33239dad2a33dz3N0~.html" ka="index_rcmd_company_42" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20181220/506bf48e3a62099d423d61d9ad505909cfcd208495d565ef66e7dff9f98764da.jpg?x-oss-process=image/resize,w_60,limit_0" alt="36氪"></p>
                                            </a>
                                            <a href="/gongsi/1009ea33239dad2a33dz3N0~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">36氪</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/12e40c0081b7fd071n192tm8EFpW.html" ka="index_rcmd_job_43" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">美食类短视频 高级编导 新东方在线</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-25K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>1-3年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/52b87ce9ef1732770XB62du7EQ~~.html" ka="index_rcmd_company_43" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20201126/2ece5bcf984e33e2ba8ea105936e75f2b8df3e971b0b84be0ebc3af5ac098b3e_s.jpg?x-oss-process=image/resize,w_60,limit_0" alt="新东方在线"></p>
                                            </a>
                                            <a href="/gongsi/52b87ce9ef1732770XB62du7EQ~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">新东方在线</span>
                                                <span class="type">在线教育</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/f0dfc9c8f4dd83b71nRy2Nu6FlZQ.html" ka="index_rcmd_job_44" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">NBA体育商业制片人</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">25-50K·16薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/2e64a887a110ea9f1nRz.html" ka="index_rcmd_company_44" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/bar/20200430/4204e9c9f200b00b77fb59d093acd281be1bd4a3bd2a63f070bdbdada9aad826.jpg?x-oss-process=image/resize,w_60,limit_0" alt="腾讯"></p>
                                            </a>
                                            <a href="/gongsi/2e64a887a110ea9f1nRz.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">腾讯</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/0909c1c3cb2841541XV-3d-4EVpR.html" ka="index_rcmd_job_45" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">流量投放</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-30K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>经验不限<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/2ee772573e1dde161Hd50tk~.html" ka="index_rcmd_company_45" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/bar/20200402/ad9cb5b320b06407da363ac95e09f985be1bd4a3bd2a63f070bdbdada9aad826.png?x-oss-process=image/resize,w_60,limit_0" alt="北京尚德机构"></p>
                                            </a>
                                            <a href="/gongsi/2ee772573e1dde161Hd50tk~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">北京尚德机构</span>
                                                <span class="type">在线教育</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                        </ul>
                        <ul class="">
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/94dc8e30b430b3791XZ629q0ElRQ.html" ka="index_rcmd_job_46" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">医疗器械研发工程师</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-30K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/3d8eeb4d48f033da1XRz2N26.html" ka="index_rcmd_company_46" class="user-info" target="_blank">
                                                <p><img src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20180705/b0d19a5d19ce52c381784422e4bf27a695b8e8d51ea8ca0f93a9209e8763e5a2.jpg" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20180705/b0d19a5d19ce52c381784422e4bf27a695b8e8d51ea8ca0f93a9209e8763e5a2.jpg?x-oss-process=image/resize,w_60,limit_0" alt="谱尼测试"></p>
                                            </a>
                                            <a href="/gongsi/3d8eeb4d48f033da1XRz2N26.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">谱尼测试</span>
                                                <span class="type">其他行业</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/434f637abf8cbe0c1XV90t25GVNS.html" ka="index_rcmd_job_47" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">高薪诚聘CRO副总经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">60-90K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/3d8eeb4d48f033da1XRz2N26.html" ka="index_rcmd_company_47" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20180705/b0d19a5d19ce52c381784422e4bf27a695b8e8d51ea8ca0f93a9209e8763e5a2.jpg?x-oss-process=image/resize,w_60,limit_0" alt="谱尼测试"></p>
                                            </a>
                                            <a href="/gongsi/3d8eeb4d48f033da1XRz2N26.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">谱尼测试</span>
                                                <span class="type">其他行业</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/0e565df0b9089e281XVy2Nm7GFVQ.html" ka="index_rcmd_job_48" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">中医医生（互联网医疗）</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-25K·13薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>大专</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/453847592ad5aa711HJ40t-0.html" ka="index_rcmd_company_48" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20160907/6ca92bb69b49b2281e6fb38a8fd8ed2fe0da977f6921b9487e34224d6d0b4481.jpg?x-oss-process=image/resize,w_60,limit_0" alt="金恪投资控股集团"></p>
                                            </a>
                                            <a href="/gongsi/453847592ad5aa711HJ40t-0.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">金恪投资控股集团</span>
                                                <span class="type">企业服务</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/e5aa7fc22c922d9f1XV93N-_E1FR.html" ka="index_rcmd_job_49" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">临床项目经理PM</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">12-22K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>1-3年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/9a7f6aa07e8944651nZ83Nu_.html" ka="index_rcmd_company_49" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/banner/2a65383857d7801445c5b8f2ecb4a903cfcd208495d565ef66e7dff9f98764da.jpg?x-oss-process=image/resize,w_60,limit_0" alt="正大天晴"></p>
                                            </a>
                                            <a href="/gongsi/9a7f6aa07e8944651nZ83Nu_.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">正大天晴</span>
                                                <span class="type">医疗健康</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/3a10df232236d20d1nB92N-0GFZR.html" ka="index_rcmd_job_50" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">临床数据经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">30-40K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>硕士</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/2c7b060913f8261f0XZ_09S8GA~~.html" ka="index_rcmd_company_50" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/upload/com/logo/20200831/9339814f77bbf123265ccad52c4bd3834157b28cac1898afda2baccf06b3237f.png?x-oss-process=image/resize,w_60,limit_0" alt="奥鸿药业"></p>
                                            </a>
                                            <a href="/gongsi/2c7b060913f8261f0XZ_09S8GA~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">奥鸿药业</span>
                                                <span class="type">制药</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/59c9ec3e98a863021n1539u-E1RV.html" ka="index_rcmd_job_51" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">mNGS病原检测医学经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-25K·14薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>硕士</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/2aeca7f0d38698021nZ409y0.html" ka="index_rcmd_company_51" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20170728/7c70a629558e4bdf23c4cf11ba385e9b0fcaee553f758e5f650c14d3900203d8.jpg?x-oss-process=image/resize,w_60,limit_0" alt="北京吉因加"></p>
                                            </a>
                                            <a href="/gongsi/2aeca7f0d38698021nZ409y0.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">北京吉因加</span>
                                                <span class="type">医疗健康</span>
                                                <span class="vline"></span>
                                                <span class="level">C轮</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/9916517baab463691XR409y1F1pW.html" ka="index_rcmd_job_52" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">高级有机合成研究员-北京</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-30K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>经验不限<span class="vline"></span>博士</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/2d72b7824b49054f0XRz2N21.html" ka="index_rcmd_company_52" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/upload/com/workfeel/20220322/7bf6f160950405e9a87b26d6cd3cde4f026084738926c00c86461661a77d7260478938dec308e828.jpg?x-oss-process=image/resize,w_60,limit_0" alt="康龙化成"></p>
                                            </a>
                                            <a href="/gongsi/2d72b7824b49054f0XRz2N21.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">康龙化成</span>
                                                <span class="type">医疗健康</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/39821b607d2e913b1XZ62N-9FFBQ.html" ka="index_rcmd_job_53" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">毒理（动物）实验室主管</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-25K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/3d8eeb4d48f033da1XRz2N26.html" ka="index_rcmd_company_53" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20180705/b0d19a5d19ce52c381784422e4bf27a695b8e8d51ea8ca0f93a9209e8763e5a2.jpg?x-oss-process=image/resize,w_60,limit_0" alt="谱尼测试"></p>
                                            </a>
                                            <a href="/gongsi/3d8eeb4d48f033da1XRz2N26.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">谱尼测试</span>
                                                <span class="type">其他行业</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/2e2e6881df2f9aba1XR83ti1FVNQ.html" ka="index_rcmd_job_54" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">分子试剂研发工程师</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-30K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>1-3年<span class="vline"></span>硕士</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/8362a1a8708854fd03Z92d61.html" ka="index_rcmd_company_54" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20170621/9ea2972c820682a34abaee5e8df8e0e4625144b83d797549c2e9df3b76783b18.jpg?x-oss-process=image/resize,w_60,limit_0" alt="卡尤迪"></p>
                                            </a>
                                            <a href="/gongsi/8362a1a8708854fd03Z92d61.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">卡尤迪</span>
                                                <span class="type">医疗健康</span>
                                                <span class="vline"></span>
                                                <span class="level">C轮</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                        </ul>
                        <ul class="">
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/e84256831cc3bc1f1XVy2d21ElRQ.html" ka="index_rcmd_job_55" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">专利市场分析师（北京）</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-30K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/b1ef0b0dc7619b591HZ409i7.html" ka="index_rcmd_company_55" class="user-info" target="_blank">
                                                <p><img src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20170505/b49cb32108226d7aa22cbd8a0f6270f1201f0e577191ae4977ee18057eee2b93.jpg" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20170505/b49cb32108226d7aa22cbd8a0f6270f1201f0e577191ae4977ee18057eee2b93.jpg?x-oss-process=image/resize,w_60,limit_0" alt="华进联合"></p>
                                            </a>
                                            <a href="/gongsi/b1ef0b0dc7619b591HZ409i7.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">华进联合</span>
                                                <span class="type">专利/商标/知识产权</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/e32a8e30e5f114631nJz09q4EVNS.html" ka="index_rcmd_job_56" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">知识产权经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">17-29K·15薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/6dab4a8e153f56c31nJ_3dy7Fg~~.html" ka="index_rcmd_company_56" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20180322/4b60c552bfe701f3e4359508ac1e3d6050c499161b871be51ef0ca652a30dea3.jpg?x-oss-process=image/resize,w_60,limit_0" alt="鸿合"></p>
                                            </a>
                                            <a href="/gongsi/6dab4a8e153f56c31nJ_3dy7Fg~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">鸿合</span>
                                                <span class="type">智能硬件</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/e8cea1b279d2da3d1XV72Ni8F1ZV.html" ka="index_rcmd_job_57" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">数据分析师</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">28-55K·15薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/49f35d19f7a5dc3d1HF53tq7Eg~~.html" ka="index_rcmd_company_57" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20190219/6d03bd46dc8125514b8e82c777938ab67d524464e8521450bf8382bd81cc65dc.png?x-oss-process=image/resize,w_60,limit_0" alt="京东振世"></p>
                                            </a>
                                            <a href="/gongsi/49f35d19f7a5dc3d1HF53tq7Eg~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">京东振世</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/79fd8f0811015d151XV92928GVJR.html" ka="index_rcmd_job_58" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">投资项目管理咨询顾问（IM/PS）</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-25K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/4fa2692f9c972f001nd72ti_ElY~.html" ka="index_rcmd_company_58" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20210105/105688fc6e38d8e001dd8ca63a1e3b71a54f7d3ca89e22df787ff870e1508ec3_s.jpg?x-oss-process=image/resize,w_60,limit_0" alt="昆仑数智"></p>
                                            </a>
                                            <a href="/gongsi/4fa2692f9c972f001nd72ti_ElY~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">昆仑数智</span>
                                                <span class="type">计算机服务</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/730ffaadb00c75991XVy39q7FldR.html" ka="index_rcmd_job_59" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">合伙人/总监，组织及人才变革管理咨询</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">70-90K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>10年以上<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/15cad93523fc4b7e33J-29u9.html" ka="index_rcmd_company_59" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20180321/b8f2b1bd4cb7ea66e635977aa2173583cfcd208495d565ef66e7dff9f98764da.jpg?x-oss-process=image/resize,w_60,limit_0" alt="毕马威"></p>
                                            </a>
                                            <a href="/gongsi/15cad93523fc4b7e33J-29u9.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">毕马威</span>
                                                <span class="type">咨询</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/8cfbc1c2c730dc1a1n1_29m4GVM~.html" ka="index_rcmd_job_60" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">法务经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-25K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/2ee772573e1dde161Hd50tk~.html" ka="index_rcmd_company_60" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/bar/20200402/ad9cb5b320b06407da363ac95e09f985be1bd4a3bd2a63f070bdbdada9aad826.png?x-oss-process=image/resize,w_60,limit_0" alt="北京尚德机构"></p>
                                            </a>
                                            <a href="/gongsi/2ee772573e1dde161Hd50tk~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">北京尚德机构</span>
                                                <span class="type">在线教育</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/f72c554a6ee2a0281XVy0924EVJR.html" ka="index_rcmd_job_61" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">资深敏捷教练</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">28-55K·14薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>10年以上<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/36869ae6585d84a91Xxz3Nu6.html" ka="index_rcmd_company_61" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20201113/30336520d0bc64cd1a4716ba84c2f503f5cca92f97e79309da04995aee3b353a_s.jpg?x-oss-process=image/resize,w_60,limit_0" alt="马上消费金融"></p>
                                            </a>
                                            <a href="/gongsi/36869ae6585d84a91Xxz3Nu6.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">马上消费金融</span>
                                                <span class="type">互联网金融</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/b342327f00988c9c1nx439S6FFdQ.html" ka="index_rcmd_job_62" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">战略分析师</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">25-50K·15薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/484a5d8d4d4a83f41nB83tS_FlI~.html" ka="index_rcmd_company_62" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/upload/com/logo/20210811/924b14eb39ef3aafece7deacf066aa4af1d0c5bf8935eb27bf0bfbc5a01d62fd.jpg?x-oss-process=image/resize,w_60,limit_0" alt="车欢欢"></p>
                                            </a>
                                            <a href="/gongsi/484a5d8d4d4a83f41nB83tS_FlI~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">车欢欢</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">D轮及以上</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/41f9ef8a14b28c8f0nB92Ny4ElM~.html" ka="index_rcmd_job_63" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">管理咨询经理——内控方向</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-30K·13薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/0139a15f633684e41XR439m5Eg~~.html" ka="index_rcmd_company_63" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/upload/com/logo/20190510/095d4b34b3abe14fc894e8a30c9d778910488a08968b447d4bfd20b1a18763a4.jpg?x-oss-process=image/resize,w_60,limit_0" alt="天职国际会计师事务所"></p>
                                            </a>
                                            <a href="/gongsi/0139a15f633684e41XR439m5Eg~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">天职国际会计师事务所</span>
                                                <span class="type">财务/审计/税务</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                        </ul>
                        <ul class="">
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/340da9137e184f0a1nB53tS8F1NR.html" ka="index_rcmd_job_64" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">京东物流招聘供应链总监</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-25K·14薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/951e8bf16e0821e40HR909i1Fw~~.html" ka="index_rcmd_company_64" class="user-info" target="_blank">
                                                <p><img src="https://img.bosszhipin.com/beijin/mcs/banner/dda5cf8c8ffc66ec166619e61e554b2acfcd208495d565ef66e7dff9f98764da.png" data-src="https://img.bosszhipin.com/beijin/mcs/banner/dda5cf8c8ffc66ec166619e61e554b2acfcd208495d565ef66e7dff9f98764da.png?x-oss-process=image/resize,w_60,limit_0" alt="京东物流"></p>
                                            </a>
                                            <a href="/gongsi/951e8bf16e0821e40HR909i1Fw~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">京东物流</span>
                                                <span class="type">物流/仓储</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/e6338cc6a17100a91XR42t2_EFNW.html" ka="index_rcmd_job_65" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">美团招聘供应链总监</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">50-80K·15薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/35c55b0fd57384791nZ80t-1EVM~.html" ka="index_rcmd_company_65" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20210425/dcc118b1cd23a89c09823e352e07dd3d2c695d20c316d909d2470f0b193f0a1a_s.jpg?x-oss-process=image/resize,w_60,limit_0" alt="美团"></p>
                                            </a>
                                            <a href="/gongsi/35c55b0fd57384791nZ80t-1EVM~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">美团</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/6fb255547c3a421a1XV83d67E1FW.html" ka="index_rcmd_job_66" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">京东物流招聘物流运营</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">16-28K·15薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/951e8bf16e0821e40HR909i1Fw~~.html" ka="index_rcmd_company_66" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/banner/dda5cf8c8ffc66ec166619e61e554b2acfcd208495d565ef66e7dff9f98764da.png?x-oss-process=image/resize,w_60,limit_0" alt="京东物流"></p>
                                            </a>
                                            <a href="/gongsi/951e8bf16e0821e40HR909i1Fw~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">京东物流</span>
                                                <span class="type">物流/仓储</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/a29d09c19f9b941a1n163t-9F1BS.html" ka="index_rcmd_job_67" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">饿了么招聘物流经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">24-30K·13薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/12e92393bff967950HNy3w~~.html" ka="index_rcmd_company_67" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20200429/2cb9f2750626ce710a1c731cb310972a0e87d15a8778c330d947906be2e52c23_s.jpg?x-oss-process=image/resize,w_60,limit_0" alt="饿了么"></p>
                                            </a>
                                            <a href="/gongsi/12e92393bff967950HNy3w~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">饿了么</span>
                                                <span class="type">O2O</span>
                                                <span class="vline"></span>
                                                <span class="level">不需要融资</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/2cb671154c37a3fd1n173927GVJR.html" ka="index_rcmd_job_68" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">闪电快车招聘物流总监</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">40-45K·14薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/3055218f1ccef2221nB53d66FA~~.html" ka="index_rcmd_company_68" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/banner/456e9378faa473da4792a4bb4d2d6d05cfcd208495d565ef66e7dff9f98764da.jpg?x-oss-process=image/resize,w_60,limit_0" alt="闪电快车"></p>
                                            </a>
                                            <a href="/gongsi/3055218f1ccef2221nB53d66FA~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">闪电快车</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">D轮及以上</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/0ff5fe6342ff76061XV93Nm6E1JS.html" ka="index_rcmd_job_69" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">小米招聘供应链经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-35K·14薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/6f1aa1d6b1d033ad33B43N0~.html" ka="index_rcmd_company_69" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/banner/71f4ee09b9a2abfb675b5c705fc46c9dcfcd208495d565ef66e7dff9f98764da.jpg?x-oss-process=image/resize,w_60,limit_0" alt="小米"></p>
                                            </a>
                                            <a href="/gongsi/6f1aa1d6b1d033ad33B43N0~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">小米</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/75181bbeccbd138a1nN_09W7FVNQ.html" ka="index_rcmd_job_70" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">小米招聘物流运营</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">25-35K·14薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/6f1aa1d6b1d033ad33B43N0~.html" ka="index_rcmd_company_70" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/banner/71f4ee09b9a2abfb675b5c705fc46c9dcfcd208495d565ef66e7dff9f98764da.jpg?x-oss-process=image/resize,w_60,limit_0" alt="小米"></p>
                                            </a>
                                            <a href="/gongsi/6f1aa1d6b1d033ad33B43N0~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">小米</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/71174e7f15742cdb0XR-3Nm1F1I~.html" ka="index_rcmd_job_71" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">京东物流招聘物流经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">14-28K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>经验不限<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/951e8bf16e0821e40HR909i1Fw~~.html" ka="index_rcmd_company_71" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/banner/dda5cf8c8ffc66ec166619e61e554b2acfcd208495d565ef66e7dff9f98764da.png?x-oss-process=image/resize,w_60,limit_0" alt="京东物流"></p>
                                            </a>
                                            <a href="/gongsi/951e8bf16e0821e40HR909i1Fw~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">京东物流</span>
                                                <span class="type">物流/仓储</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/bf839fa8fc607c351nx72N-0FVNQ.html" ka="index_rcmd_job_72" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">京东物流招聘供应链经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-25K·15薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>1-3年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/951e8bf16e0821e40HR909i1Fw~~.html" ka="index_rcmd_company_72" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/banner/dda5cf8c8ffc66ec166619e61e554b2acfcd208495d565ef66e7dff9f98764da.png?x-oss-process=image/resize,w_60,limit_0" alt="京东物流"></p>
                                            </a>
                                            <a href="/gongsi/951e8bf16e0821e40HR909i1Fw~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">京东物流</span>
                                                <span class="type">物流/仓储</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                        </ul>
                        <ul class="">
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/ce734b196ba1c0261XV_3N-4FlNQ.html" ka="index_rcmd_job_73" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">寻源经理--半导体</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-30K·15薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/a67b361452e384e71XV82N4~.html" ka="index_rcmd_company_73" class="user-info" target="_blank">
                                                <p><img src="https://img.bosszhipin.com/beijin/upload/com/logo/20210525/77d60eae41e48b90df64951371a7a07a19f97e2c258c6cead07beaf11928d91b.png" data-src="https://img.bosszhipin.com/beijin/upload/com/logo/20210525/77d60eae41e48b90df64951371a7a07a19f97e2c258c6cead07beaf11928d91b.png?x-oss-process=image/resize,w_60,limit_0" alt="今日头条"></p>
                                            </a>
                                            <a href="/gongsi/a67b361452e384e71XV82N4~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">今日头条</span>
                                                <span class="type">移动互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">D轮及以上</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/70e852a96c3ae7931nN_3tm4FVRW.html" ka="index_rcmd_job_74" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">采购总监</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-40K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>经验不限<span class="vline"></span>学历不限</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/81db0b50491a53a533F53to~.html" ka="index_rcmd_company_74" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/banner/9ac98d9472869779b026f43947537af0cfcd208495d565ef66e7dff9f98764da.jpg?x-oss-process=image/resize,w_60,limit_0" alt="星汉博纳医药"></p>
                                            </a>
                                            <a href="/gongsi/81db0b50491a53a533F53to~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">星汉博纳医药</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">D轮及以上</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/5fce2626e0087ed01XV82Nq8FldV.html" ka="index_rcmd_job_75" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">蔬果采购经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-25K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>3-5年<span class="vline"></span>大专</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/308b65efbdf6400a1nd92N-8ElM~.html" ka="index_rcmd_company_75" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20210121/8fb6313906856cdb0d7266000988ff2a727e7e8e34a037604db8e4ba9030afc5_s.jpg?x-oss-process=image/resize,w_60,limit_0" alt="美菜"></p>
                                            </a>
                                            <a href="/gongsi/308b65efbdf6400a1nd92N-8ElM~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">美菜</span>
                                                <span class="type">电子商务</span>
                                                <span class="vline"></span>
                                                <span class="level">D轮及以上</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/d041aee862d0faad1XRz39i_EFdW.html" ka="index_rcmd_job_76" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">美团优选-休食统采专家</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">30-50K·13薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/b633a34f787d94f21nZ_0929E1I~.html" ka="index_rcmd_company_76" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/upload/com/logo/20210420/f9bff82be5bd8c7fd799be43dae6079577c47d994acd32f441228b04529816d2.png?x-oss-process=image/resize,w_60,limit_0" alt="美团"></p>
                                            </a>
                                            <a href="/gongsi/b633a34f787d94f21nZ_0929E1I~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">美团</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/de8f9c7931980a9e1nx709q5GVVV.html" ka="index_rcmd_job_77" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">采购高级经理（母婴）</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-22K·14薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/fd2ac4a4ea593bb41XZ609U~.html" ka="index_rcmd_company_77" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img2.bosszhipin.com/mcs/chatphoto/20160328/04262ecfa9195677b8de595d491d1a28b1d36d3491223e4bb6db711b34c37ccd.jpg?x-oss-process=image/resize,w_60,limit_0" alt="国美电器"></p>
                                            </a>
                                            <a href="/gongsi/fd2ac4a4ea593bb41XZ609U~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">国美电器</span>
                                                <span class="type">其他行业</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/28a51195d6fc5f331XVz39S-GFdW.html" ka="index_rcmd_job_78" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">商品运营</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">15-25K·13薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>大专</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/38bf2d9641330eef1XZ-2A~~.html" ka="index_rcmd_company_78" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/bar/20190702/6261090e70ee610815076fbaddae1a74be1bd4a3bd2a63f070bdbdada9aad826.jpg?x-oss-process=image/resize,w_60,limit_0" alt="叮当快药"></p>
                                            </a>
                                            <a href="/gongsi/38bf2d9641330eef1XZ-2A~~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">叮当快药</span>
                                                <span class="type">互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">C轮</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/a942f455bc5941c71XV539m5EVVV.html" ka="index_rcmd_job_79" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">跨境电商采销总监、专家、专员</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-40K·14薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/fd2ac4a4ea593bb41XZ609U~.html" ka="index_rcmd_company_79" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img2.bosszhipin.com/mcs/chatphoto/20160328/04262ecfa9195677b8de595d491d1a28b1d36d3491223e4bb6db711b34c37ccd.jpg?x-oss-process=image/resize,w_60,limit_0" alt="国美电器"></p>
                                            </a>
                                            <a href="/gongsi/fd2ac4a4ea593bb41XZ609U~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">国美电器</span>
                                                <span class="type">其他行业</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/d3a39ec969476c911nR50ty1FFVR.html" ka="index_rcmd_job_80" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">供应链质量项目经理</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">25-50K</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/980f48937a13792b1nd63d0~.html" ka="index_rcmd_company_80" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20190408/c23f08b24983fffa26a3a8ba19a463523cc05a6873981b80bf124ddd6c45f629_s.jpg?x-oss-process=image/resize,w_60,limit_0" alt="滴滴出行"></p>
                                            </a>
                                            <a href="/gongsi/980f48937a13792b1nd63d0~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">滴滴出行</span>
                                                <span class="type">移动互联网</span>
                                                <span class="vline"></span>
                                                <span class="level">D轮及以上</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                                <li>
                                    <div class="sub-li">
                                        <a href="/job_detail/7c12fa0895df667c1XV40t-5EVBX.html" ka="index_rcmd_job_81" class="job-info" target="_blank">
                                            <div class="sub-li-top">
                                                <p class="name">国际商务经理 (MJ011032)</p>
                                                <div class="guide-app-download-icon"></div>
                                                <p class="salary">20-40K·16薪</p>
                                            </div>
                                            <p class="job-text">北京<span class="vline"></span>5-10年<span class="vline"></span>本科</p>
                                        </a>
                                        <div class="sub-li-bottom">
                                            <a href="/gongsi/6b92816167c0cc421XR-3N26Fls~.html" ka="index_rcmd_company_81" class="user-info" target="_blank">
                                                <p><img src="" data-src="https://img.bosszhipin.com/beijin/icon/894ce6fa7e58d64d57e7f22d2f3a9d18afa7fcceaa24b8ea28f56f1bb14732c0.png?x-oss-process=image/resize,w_60,limit_0" alt="星网锐捷"></p>
                                            </a>
                                            <a href="/gongsi/6b92816167c0cc421XR-3N26Fls~.html" class="sub-li-bottom-commany-info" target="_blank">
                                                <span class="name">星网锐捷</span>
                                                <span class="type">通信/网络设备</span>
                                                <span class="vline"></span>
                                                <span class="level">已上市</span>
                                            </a>
                                        </div>
                                    </div>
                                </li>
                        </ul>
                <p class="common-tab-more"><a class="btn btn-outline" href="/c101010100/" ka="open_joblist">查看更多</a></p>
            </div>
            <!--公司tab列表-->
                <div class="common-tab-box hot-company-box">
                    <div class="box-title">热门企业</div>
                    <ul class="cur">
                            <li>
                                <!-- 公司信息 -->
                                <a class="company-info-top" href="/gongsi/b633a34f787d94f21nZ_0929E1I~.html" ka="index_rcmd_companylogo_1_custompage" target="_blank">
                                    <div class="company-img">
                                        <img src="https://img.bosszhipin.com/beijin/upload/com/logo/20210420/f9bff82be5bd8c7fd799be43dae6079577c47d994acd32f441228b04529816d2.png?x-oss-process=image/resize,w_100,limit_0" data-src="https://img.bosszhipin.com/beijin/upload/com/logo/20210420/f9bff82be5bd8c7fd799be43dae6079577c47d994acd32f441228b04529816d2.png?x-oss-process=image/resize,w_100,limit_0" alt="美团">
                                    </div>
                                    <div class="company-info">
                                        <h3>美团</h3>
                                        <p>已上市<span class="vline"></span>10000人以上<span class="vline"></span>互联网</p>
                                    </div>
                                </a>
                                <!-- 职位列表 -->
                                    <ul class="company-job-list">
                                            <li class="company-job-item">
                                                <a href="/job_detail/7418b543dd0210111nVz3N-9ElZX.html" ka="index_rcmd_company_job_119720247_12580030" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">共享单车运维</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">6-11K</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 朝阳区 四惠</span><span>1-3年</span><span>学历不限</span></p>
                                                </a>
                                            </li>
                                            <li class="company-job-item">
                                                <a href="/job_detail/53f44c492d1d97500XR-3ti7E1Q~.html" ka="index_rcmd_company_job_60455636_12580030" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">产品经理</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">15-20K·15薪</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 朝阳区 望京</span><span>1-3年</span><span>本科</span></p>
                                                </a>
                                            </li>
                                            <li class="company-job-item">
                                                <a href="/job_detail/9bb85ca05cacf10b1n1509S7FVBX.html" ka="index_rcmd_company_job_193896527_12580030" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">美团优选策略产品实习生</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">200-250元/天</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 朝阳区 望京</span><span>5天/周</span><span>4个月</span><span>本科</span></p>
                                                </a>
                                            </li>
                                    </ul>
                                <a href="/gongsi/b633a34f787d94f21nZ_0929E1I~.html" target="_blank" ka="index_rcmd_company_1_custompage" class="more-job-btn">查看更多职位</a>
                            </li>
                            <li>
                                <!-- 公司信息 -->
                                <a class="company-info-top" href="/gongsi/f4b44567ac3360131nV829W9GFI~.html" ka="index_rcmd_companylogo_2_custompage" target="_blank">
                                    <div class="company-img">
                                        <img src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20201016/5790ff5c3a87201600523676ec508543c38aab861bd796eedc794b34e3418aec_s.jpg?x-oss-process=image/resize,w_100,limit_0" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20201016/5790ff5c3a87201600523676ec508543c38aab861bd796eedc794b34e3418aec_s.jpg?x-oss-process=image/resize,w_100,limit_0" alt="中安建培">
                                    </div>
                                    <div class="company-info">
                                        <h3>中安建培</h3>
                                        <p>不需要融资<span class="vline"></span>1000-9999人<span class="vline"></span>培训机构</p>
                                    </div>
                                </a>
                                <!-- 职位列表 -->
                                    <ul class="company-job-list">
                                            <li class="company-job-item">
                                                <a href="/job_detail/4aa15b8bf00898251n1y09-8EVRQ.html" ka="index_rcmd_company_job_198821160_11608080" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">均薪1万/接受小白/不用外出/提供精准资源</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">9-14K</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 昌平区 天通苑</span><span>经验不限</span><span>中专/中技</span></p>
                                                </a>
                                            </li>
                                            <li class="company-job-item">
                                                <a href="/job_detail/2b77e7757aa96db51XR_2N-7EFJR.html" ka="index_rcmd_company_job_205326001_11608080" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">纯微信销售/不打电话/五险一金</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">17-30K</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 丰台区 六里桥</span><span>经验不限</span><span>学历不限</span></p>
                                                </a>
                                            </li>
                                            <li class="company-job-item">
                                                <a href="/job_detail/0819869346ce62621XR_2Ni9ElZY.html" ka="index_rcmd_company_job_205350248_11608080" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">浮动底薪5500-8500/超高提点/年薪35万＋</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">17-30K</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 昌平区 回龙观</span><span>经验不限</span><span>学历不限</span></p>
                                                </a>
                                            </li>
                                    </ul>
                                <a href="/gongsi/f4b44567ac3360131nV829W9GFI~.html" target="_blank" ka="index_rcmd_company_2_custompage" class="more-job-btn">查看更多职位</a>
                            </li>
                            <li>
                                <!-- 公司信息 -->
                                <a class="company-info-top" href="/gongsi/1edcbdb0e9ec98191nZ_0t65F1Y~.html" ka="index_rcmd_companylogo_3_custompage" target="_blank">
                                    <div class="company-img">
                                        <img src="https://c-res.zhipin.com/jrs/ce16a8349d0cee84b271350a139213b3.jpg?x-oss-process=image/resize,w_100,limit_0" data-src="https://c-res.zhipin.com/jrs/ce16a8349d0cee84b271350a139213b3.jpg?x-oss-process=image/resize,w_100,limit_0" alt="京东物流-华北区域">
                                    </div>
                                    <div class="company-info">
                                        <h3>京东物流-华北区域</h3>
                                        <p>已上市<span class="vline"></span>10000人以上<span class="vline"></span>物流/仓储</p>
                                    </div>
                                </a>
                                <!-- 职位列表 -->
                                    <ul class="company-job-list">
                                            <li class="company-job-item">
                                                <a href="/job_detail/6c60451f65fbaaff0Xx40ty9EVE~.html" ka="index_rcmd_company_job_68291013_12593474" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">客户管理岗</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">11-20K·14薪</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 通州区 次渠</span><span>5-10年</span><span>大专</span></p>
                                                </a>
                                            </li>
                                            <li class="company-job-item">
                                                <a href="/job_detail/fb73a9d111afb6881nN82dm8GFJR.html" ka="index_rcmd_company_job_176241801_12593474" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">长期招聘日结包装工</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">6-9K</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 大兴区 庞各庄</span><span>经验不限</span><span>初中及以下</span></p>
                                                </a>
                                            </li>
                                            <li class="company-job-item">
                                                <a href="/job_detail/307e6836f0d447091n193dW5GFFQ.html" ka="index_rcmd_company_job_197684830_12593474" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">纯坐岗包装贴签员</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">7-10K</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 大兴区 黄村</span><span>经验不限</span><span>学历不限</span></p>
                                                </a>
                                            </li>
                                    </ul>
                                <a href="/gongsi/1edcbdb0e9ec98191nZ_0t65F1Y~.html" target="_blank" ka="index_rcmd_company_3_custompage" class="more-job-btn">查看更多职位</a>
                            </li>
                            <li>
                                <!-- 公司信息 -->
                                <a class="company-info-top" href="/gongsi/8372b71df55405071nJ52Nu9.html" ka="index_rcmd_companylogo_4_custompage" target="_blank">
                                    <div class="company-img">
                                        <img src="https://img.bosszhipin.com/beijin/upload/com/workfeel/20220221/7bf6f160950405e936172f3da469a5b73527f92c14bca24666362c7492a15abeb44f891ff2bfbc6b.jpg?x-oss-process=image/resize,w_100,limit_0" data-src="https://img.bosszhipin.com/beijin/upload/com/workfeel/20220221/7bf6f160950405e936172f3da469a5b73527f92c14bca24666362c7492a15abeb44f891ff2bfbc6b.jpg?x-oss-process=image/resize,w_100,limit_0" alt="知乎">
                                    </div>
                                    <div class="company-info">
                                        <h3>知乎</h3>
                                        <p>已上市<span class="vline"></span>1000-9999人<span class="vline"></span>移动互联网</p>
                                    </div>
                                </a>
                                <!-- 职位列表 -->
                                    <ul class="company-job-list">
                                            <li class="company-job-item">
                                                <a href="/job_detail/9c3b41a9ae0869cb1XR639W0GFZW.html" ka="index_rcmd_company_job_200489846_163360" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">Android 开发工程师</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">25-50K·15薪</p>
                                                    </div>
                                                    <p class="job-text"><span>北京</span><span>经验不限</span><span>学历不限</span></p>
                                                </a>
                                            </li>
                                            <li class="company-job-item">
                                                <a href="/job_detail/4a900c6376912c311nJ92t-1E1BX.html" ka="index_rcmd_company_job_167128327_163360" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">Android 开发工程师（社区）</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">50-70K·15薪</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 海淀区 五道口</span><span>经验不限</span><span>学历不限</span></p>
                                                </a>
                                            </li>
                                            <li class="company-job-item">
                                                <a href="/job_detail/6edd0e3969c60eda1nx82ti5ElFQ.html" ka="index_rcmd_company_job_186154230_163360" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">测试开发工程师</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">30-60K·15薪</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 海淀区 五道口</span><span>经验不限</span><span>本科</span></p>
                                                </a>
                                            </li>
                                    </ul>
                                <a href="/gongsi/8372b71df55405071nJ52Nu9.html" target="_blank" ka="index_rcmd_company_4_custompage" class="more-job-btn">查看更多职位</a>
                            </li>
                            <li>
                                <!-- 公司信息 -->
                                <a class="company-info-top" href="/gongsi/3ee93e46b66656631nV929i8GQ~~.html" ka="index_rcmd_companylogo_5_custompage" target="_blank">
                                    <div class="company-img">
                                        <img src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20171029/5651a54b0e2a5a4d5886f943f2ea3f693caeaa7db720712ad677305059425655.jpg?x-oss-process=image/resize,w_100,limit_0" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20171029/5651a54b0e2a5a4d5886f943f2ea3f693caeaa7db720712ad677305059425655.jpg?x-oss-process=image/resize,w_100,limit_0" alt="MetLife北京">
                                    </div>
                                    <div class="company-info">
                                        <h3>MetLife北京</h3>
                                        <p>已上市<span class="vline"></span>1000-9999人<span class="vline"></span>互联网金融</p>
                                    </div>
                                </a>
                                <!-- 职位列表 -->
                                    <ul class="company-job-list">
                                            <li class="company-job-item">
                                                <a href="/job_detail/9a55461c50d213981n1809W4FVFV.html" ka="index_rcmd_company_job_196885535_1170519" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">团队管理经理</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">15-30K</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 朝阳区 双井</span><span>3-5年</span><span>本科</span></p>
                                                </a>
                                            </li>
                                            <li class="company-job-item">
                                                <a href="/job_detail/35a20ac08d47fbdc1nN_0ti4GVRS.html" ka="index_rcmd_company_job_175955962_1170519" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">培训运营管理</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">15-30K·13薪</p>
                                                    </div>
                                                    <p class="job-text"><span>北京</span><span>1-3年</span><span>大专</span></p>
                                                </a>
                                            </li>
                                            <li class="company-job-item">
                                                <a href="/job_detail/1845bbf459dbf9531XR53tq4FlpS.html" ka="index_rcmd_company_job_203575682_1170519" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">🇨🇳5小时半天班➕客服专员➕七险一金</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">7-8K</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 朝阳区 酒仙桥</span><span>经验不限</span><span>学历不限</span></p>
                                                </a>
                                            </li>
                                    </ul>
                                <a href="/gongsi/3ee93e46b66656631nV929i8GQ~~.html" target="_blank" ka="index_rcmd_company_5_custompage" class="more-job-btn">查看更多职位</a>
                            </li>
                            <li>
                                <!-- 公司信息 -->
                                <a class="company-info-top" href="/gongsi/c7cf8f5b4b7e1a271nZ42Ng~.html" ka="index_rcmd_companylogo_6_custompage" target="_blank">
                                    <div class="company-img">
                                        <img src="https://img.bosszhipin.com/beijin/upload/com/workfeel/20210525/91c317baff968ca4eb1498357773a20bad13fbc25a880247c035c5ab6a6a144e.jpg?x-oss-process=image/resize,w_100,limit_0" data-src="https://img.bosszhipin.com/beijin/upload/com/workfeel/20210525/91c317baff968ca4eb1498357773a20bad13fbc25a880247c035c5ab6a6a144e.jpg?x-oss-process=image/resize,w_100,limit_0" alt="开课吧">
                                    </div>
                                    <div class="company-info">
                                        <h3>开课吧</h3>
                                        <p>B轮<span class="vline"></span>1000-9999人<span class="vline"></span>互联网</p>
                                    </div>
                                </a>
                                <!-- 职位列表 -->
                                    <ul class="company-job-list">
                                            <li class="company-job-item">
                                                <a href="/job_detail/251d05ebe01d16461nJ63d-_F1ZW.html" ka="index_rcmd_company_job_160622746_12235" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">测试工程师</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">15-30K</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 海淀区 上地</span><span>3-5年</span><span>本科</span></p>
                                                </a>
                                            </li>
                                            <li class="company-job-item">
                                                <a href="/job_detail/8a0819592e6031ce1nxy2dq0FFZR.html" ka="index_rcmd_company_job_188279441_12235" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">运维兼职讲师</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">20-25K</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 海淀区 上地</span><span>3-5年</span><span>学历不限</span></p>
                                                </a>
                                            </li>
                                            <li class="company-job-item">
                                                <a href="/job_detail/2fed1dd9e9298f221nN43tm4ElVX.html" ka="index_rcmd_company_job_172545277_12235" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">学科运营</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">15-30K</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 海淀区 上地</span><span>1-3年</span><span>本科</span></p>
                                                </a>
                                            </li>
                                    </ul>
                                <a href="/gongsi/c7cf8f5b4b7e1a271nZ42Ng~.html" target="_blank" ka="index_rcmd_company_6_custompage" class="more-job-btn">查看更多职位</a>
                            </li>
                            <li>
                                <!-- 公司信息 -->
                                <a class="company-info-top" href="/gongsi/2e64a887a110ea9f1nRz.html" ka="index_rcmd_companylogo_7_custompage" target="_blank">
                                    <div class="company-img">
                                        <img src="https://img.bosszhipin.com/beijin/mcs/bar/20200430/4204e9c9f200b00b77fb59d093acd281be1bd4a3bd2a63f070bdbdada9aad826.jpg?x-oss-process=image/resize,w_100,limit_0" data-src="https://img.bosszhipin.com/beijin/mcs/bar/20200430/4204e9c9f200b00b77fb59d093acd281be1bd4a3bd2a63f070bdbdada9aad826.jpg?x-oss-process=image/resize,w_100,limit_0" alt="腾讯">
                                    </div>
                                    <div class="company-info">
                                        <h3>腾讯</h3>
                                        <p>不需要融资<span class="vline"></span>10000人以上<span class="vline"></span>互联网</p>
                                    </div>
                                </a>
                                <!-- 职位列表 -->
                                    <ul class="company-job-list">
                                            <li class="company-job-item">
                                                <a href="/job_detail/2f790765afd807ef0XB93dW_EVY~.html" ka="index_rcmd_company_job_64768214_109" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">腾讯新闻 GO研发工程师</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">25-50K</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 海淀区 西北旺</span><span>3-5年</span><span>本科</span></p>
                                                </a>
                                            </li>
                                            <li class="company-job-item">
                                                <a href="/job_detail/4addf65498a69e051nN50961EFZZ.html" ka="index_rcmd_company_job_173838049_109" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">Golang开发工程师</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">25-40K·16薪</p>
                                                    </div>
                                                    <p class="job-text"><span>北京</span><span>经验不限</span><span>本科</span></p>
                                                </a>
                                            </li>
                                            <li class="company-job-item">
                                                <a href="/job_detail/9f7bd5b1d84494fd0Xx83d-_F1Q~.html" ka="index_rcmd_company_job_68662276_109" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">数据采集工程师</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">6-11K·14薪</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 昌平区 南邵</span><span>1-3年</span><span>大专</span></p>
                                                </a>
                                            </li>
                                    </ul>
                                <a href="/gongsi/2e64a887a110ea9f1nRz.html" target="_blank" ka="index_rcmd_company_7_custompage" class="more-job-btn">查看更多职位</a>
                            </li>
                            <li>
                                <!-- 公司信息 -->
                                <a class="company-info-top" href="/gongsi/33e052361693f8371nF-3d25.html" ka="index_rcmd_companylogo_8_custompage" target="_blank">
                                    <div class="company-img">
                                        <img src="https://img.bosszhipin.com/beijin/mcs/bar/20191129/3cdf5ba2149e309b38868b62ae9c22cabe1bd4a3bd2a63f070bdbdada9aad826.jpg?x-oss-process=image/resize,w_100,limit_0" data-src="https://img.bosszhipin.com/beijin/mcs/bar/20191129/3cdf5ba2149e309b38868b62ae9c22cabe1bd4a3bd2a63f070bdbdada9aad826.jpg?x-oss-process=image/resize,w_100,limit_0" alt="京东集团">
                                    </div>
                                    <div class="company-info">
                                        <h3>京东集团</h3>
                                        <p>已上市<span class="vline"></span>10000人以上<span class="vline"></span>电子商务</p>
                                    </div>
                                </a>
                                <!-- 职位列表 -->
                                    <ul class="company-job-list">
                                            <li class="company-job-item">
                                                <a href="/job_detail/c4324cbf55e5e0f31XBy3Nu_FlI~.html" ka="index_rcmd_company_job_24876260_154604" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">项目管理岗</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">13-21K</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 通州区 次渠</span><span>3-5年</span><span>本科</span></p>
                                                </a>
                                            </li>
                                            <li class="company-job-item">
                                                <a href="/job_detail/789434c1383c10061nJ92tW6ElVQ.html" ka="index_rcmd_company_job_167187270_154604" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">项目管理</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">15-20K</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 大兴区 亦庄</span><span>5-10年</span><span>本科</span></p>
                                                </a>
                                            </li>
                                            <li class="company-job-item">
                                                <a href="/job_detail/6cf629b1ccc1889b1nJ42NW9FVNU.html" ka="index_rcmd_company_job_162380514_154604" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">产品经理</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">3-4K</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 大兴区 亦庄</span><span>经验不限</span><span>本科</span></p>
                                                </a>
                                            </li>
                                    </ul>
                                <a href="/gongsi/33e052361693f8371nF-3d25.html" target="_blank" ka="index_rcmd_company_8_custompage" class="more-job-btn">查看更多职位</a>
                            </li>
                            <li>
                                <!-- 公司信息 -->
                                <a class="company-info-top" href="/gongsi/8531731ebc286f821nV73t4~.html" ka="index_rcmd_companylogo_9_custompage" target="_blank">
                                    <div class="company-img">
                                        <img src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20181130/5a7f5dfbd4b0ca719f5955ae5c476947cfcd208495d565ef66e7dff9f98764da.jpg?x-oss-process=image/resize,w_100,limit_0" data-src="https://img.bosszhipin.com/beijin/mcs/chatphoto/20181130/5a7f5dfbd4b0ca719f5955ae5c476947cfcd208495d565ef66e7dff9f98764da.jpg?x-oss-process=image/resize,w_100,limit_0" alt="京北方">
                                    </div>
                                    <div class="company-info">
                                        <h3>京北方</h3>
                                        <p>已上市<span class="vline"></span>10000人以上<span class="vline"></span>互联网金融</p>
                                    </div>
                                </a>
                                <!-- 职位列表 -->
                                    <ul class="company-job-list">
                                            <li class="company-job-item">
                                                <a href="/job_detail/33321cb6f68d7dbc1XR73tu_FlFT.html" ka="index_rcmd_company_job_201562633_11153" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">测试工程师</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">10-15K</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 海淀区 西二旗</span><span>经验不限</span><span>本科</span></p>
                                                </a>
                                            </li>
                                            <li class="company-job-item">
                                                <a href="/job_detail/b110dcc93262a55f1XR73tS8EVNZ.html" ka="index_rcmd_company_job_201591119_11153" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">接口测试</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">12-20K</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 海淀区 魏公村</span><span>3-5年</span><span>本科</span></p>
                                                </a>
                                            </li>
                                            <li class="company-job-item">
                                                <a href="/job_detail/ee63141d6555423c1XR73tu4GVpR.html" ka="index_rcmd_company_job_201565981_11153" class="job-info" target="_blank">
                                                    <div class="job-info-top">
                                                        <p class="name">软件测试工程师（各区可推荐）</p>
                                                        <div class="guide-app-download-icon"></div>
                                                        <p class="salary">11-18K</p>
                                                    </div>
                                                    <p class="job-text"><span>北京 海淀区 西北旺</span><span>经验不限</span><span>大专</span></p>
                                                </a>
                                            </li>
                                    </ul>
                                <a href="/gongsi/8531731ebc286f821nV73t4~.html" target="_blank" ka="index_rcmd_company_9_custompage" class="more-job-btn">查看更多职位</a>
                            </li>
                    </ul>
                    <p class="common-tab-more"><a class="btn btn-outline" href="/gongsi/_zzz_c101010100/" ka="open_brand">查看更多</a></p>
                </div>
        </div>
    </div>
'''
bsp = BeautifulSoup(html, 'lxml')
# trs = bsp.findAll("tr")
# for tr in trs:
#     print(tr)
print(bsp.stripped_strings)
for s in bsp.stripped_strings:
    print(s)
