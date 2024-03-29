
class SnowmanConstants:
    """CONSTANTS FOR CELL IDENTIFICATION"""
    WALL_CELL = 0
    OUT_OFF_GRID_CELL = 0
    SMALL_BALL_CELL = 1
    MEDIUM_BALL_CELL = 2
    SMALL_BALL_ON_MEDIUM_BALL_CELL= 3
    LARGE_BALL_CELL = 4
    SMALL_BALL_ON_LARGE_BALL_CELL = 5
    MEDIUM_BALL_ON_LARGE_BALL_CELL = 6
    FULL_SNOW_MAN_CELL= 7
    GRASS_CELL = 8
    SNOW_CELL = 9
    CHARACTER_ON_SNOW_CELL = 10
    CHARACTER_ON_GRASS_CELL = 11

    """CONSTANTS FOR TOKEN IDENTIFICATION"""
    WALL_TOKEN = '#'
    OUT_OFF_GRID_TOKEN = 'x'
    SMALL_BALL_TOKEN = '1'
    MEDIUM_BALL_TOKEN = '2'
    SMALL_BALL_ON_MEDIUM_BALL_TOKEN = '3'
    LARGE_BALL_TOKEN = '4'
    SMALL_BALL_ON_LARGE_BALL_TOKEN = '5'
    MEDIUM_BALL_ON_LARGE_BALL_TOKEN = '6'
    FULL_SNOW_MAN_TOKEN= '7'
    GRASS_TOKEN = ','
    SNOW_TOKEN = '.'
    CHARACTER_ON_SNOW_TOKEN = 'p'
    CHARACTER_ON_GRASS_TOKEN = 'q'
    CHARACTER_LEAVE_CELL = 'pq'

    """DEFINE REWARDS"""
    error=0
    tonto=0
    cami=-3
    cim=250
    convertir=40
    bingo=500
    arrastra=20

    actions=[
        #me->wall->wall         me->wall->small         me->wall->medium       me->wall->sm_on_md     me->wall->large       me->wall->sm_on_lg     me->wall->md_on_lg     me->wall->snowman       me->wall->grass         me->wall->snow          
        [[None,None,None,error],[None,None,None,error],[None,None,None,error],[None,None,None,error],[None,None,None,error],[None,None,None,error],[None,None,None,error],[None,None,None,error],[None,None,None,error],[None,None,None,error]],
         #me->small->wall        me->small->small      me->small->medium   me->small->sm_on_md     me->small->large  me->small->sm_on_lg   me->small->md_on_lg   me->small->snowman      me->small->grass   me->small->snow          
        [[None,None,None,error],[None,None,None,tonto],['pq', 11, 3, cim],[None,None,None,tonto],['pq', 11, 5, cim],[None,None,None,tonto],['pq', 11, 7, bingo],[None,None,None,tonto],['pq', 11, 1, arrastra],['pq',11,2,convertir]],
        #me->medium->wall        me->medium->small      me->medium->medium   me->medium->sm_on_md     me->medium->large  me->medium->sm_on_lg   me->medium->md_on_lg   me->medium->snowman   me->medium->grass  me->medium->snow          
        [[None,None,None,error],[None,None,None,tonto],[None,None,None,tonto],[None,None,None,tonto],['pq',11, 6,cim],[None,None,None,tonto],[None,None,None,tonto],[None,None,None,tonto],['pq',11,2,arrastra],['pq',11,4,convertir]],
        #me->sm_md->wall        me->sm_md->small      me->sm_md->medium       me->sm_md->sm_on_md     me->sm_md->large     me->sm_md->sm_on_lg    me->sm_md->md_on_lg     me->sm_md->snowman     me->sm_md->grass me->sm_md->snow          
        [[None,None,None,error],[None,None,None,tonto],[None,None,None,tonto],[None,None,None,tonto],[None,None,None,tonto],[None,None,None,tonto],[None,None,None,tonto],[None,None,None,tonto],[None,2,1,-cim], [ None,2,2,-cim]],
        #me->large->wall        me->large->small       me->large->medium       me->large->sm_on_md    me->large->large      me->large->sm_on_lg    me->large->md_on_lg     me->large->snowman    me->large->grass    me->large->snow          
        [[None,None,None,error],[None,None,None,tonto],[None,None,None,tonto],[None,None,None,tonto],[None,None,None,tonto],[None,None,None,tonto],[None,None,None,tonto],[None,None,None,tonto],['pq',11, 4, arrastra],['pq',11,4,arrastra]],
        #me->sm_on_lg->wall      me->sm_on_lg->small    me->sm_on_lg->medium   me->sm_on_lg->sm_on_md    me->sm_on_lg->large   me->sm_on_lg->sm_on_lg  me->sm_on_lg->md_on_lg    me->sm_on_lg->snowman  me->sm_on_lg->grass   me->sm_on_lg->snow          
        [[None,None,None,error],[None,None,None,tonto],[None,None,None,tonto],[None, None, None, tonto],[None,None,None,tonto],[None,None,None, tonto ], [None,None,None,tonto],[None,None,None,tonto],[None,  4, 1, -cim ], [None, 4, 2, -cim]],
        #me->md_on_lg->wall      me->md_on_lg->small   me->md_on_lg->medium   me->md_on_lg->sm_on_md  me->md_on_lg->large   me->md_on_lg->sm_on_lg  me->md_on_lg->md_on_lg  me->md_on_lg->snowman  me->md_on_lg->grass   me->md_on_lg->snow          
        [[None,None,None,error],[None,None,None,tonto],[None,None,None,tonto],[None,None,None,tonto],[None,None,None,tonto],[None,None,None,tonto], [None,None,None,tonto],  [None,None,None,tonto],[None,4,2,-cim],[None,4,4,-cim]],   
        #me->snowman->wall      me->snowman->small     me->snowman->medium   me->snowman->sm_on_md  me->snowman->large     me->snowman->sm_on_lg  me->snowman->md_on_lg   me->snowman->snowman   me->snowman->grass     me->snowman->snow          
        [[None,None,None,error],[None,None,None,error],[None,None,None,error],[None,None,None,error],[None,None,None,error],[None,None,None,error],[None,None,None,error],[None,None,None,error],[None,None,None,error],[None,None,None,error]],
        #me->grass->wall      me->grass->small     me->grass->medium   me->grass->sm_on_md  me->grass->large  me->grass->sm_on_lg  me->grass->md_on_lg  me->grass->snowman   me->grass->grass     me->grass->snow          
        [['pq',11,None,cami],['pq',11,None,cami],['pq',11,None,cami],['pq',11,None,cami],['pq',11,None,cami], ['pq',11,None,cami], ['pq',11,None,cami],['pq',11,None,cami],['pq',11,None,cami],['pq',11,None,cami]],
        #me->snow->wall      me->snow->small       me->snow->medium   me->snow->sm_on_md  me->snow->large     me->snow->sm_on_lg  me->snow->md_on_lg  me->snow->snowman    me->snow->grass     me->snow->snow          
        [['pq',10,None,cami],['pq',10,None,cami],['pq',10,None,cami],['pq',10,None,cami],['pq',10,None,cami],['pq',10,None,cami],['pq',10,None,cami],['pq',10,None,cami],['pq',10,None,cami],['pq',10,None,cami]]
    ]

    @staticmethod
    def get_cell_codes_array():
        return [
            SnowmanConstants.WALL_CELL,
            SnowmanConstants.OUT_OFF_GRID_CELL,
            SnowmanConstants.SMALL_BALL_CELL,
            SnowmanConstants.MEDIUM_BALL_CELL,
            SnowmanConstants.SMALL_BALL_ON_MEDIUM_BALL_CELL,
            SnowmanConstants.LARGE_BALL_CELL,
            SnowmanConstants.SMALL_BALL_ON_LARGE_BALL_CELL,
            SnowmanConstants.MEDIUM_BALL_ON_LARGE_BALL_CELL,
            SnowmanConstants.FULL_SNOW_MAN_CELL,
            SnowmanConstants.GRASS_CELL,
            SnowmanConstants.SNOW_CELL,
            SnowmanConstants.CHARACTER_ON_SNOW_CELL,
            SnowmanConstants.CHARACTER_ON_GRASS_CELL
        ]
    
    @staticmethod
    def get_tokens_array():
        return [
            SnowmanConstants.WALL_TOKEN,
            SnowmanConstants.OUT_OFF_GRID_TOKEN,
            SnowmanConstants.SMALL_BALL_TOKEN,
            SnowmanConstants.MEDIUM_BALL_TOKEN,
            SnowmanConstants.SMALL_BALL_ON_MEDIUM_BALL_TOKEN,
            SnowmanConstants.LARGE_BALL_TOKEN,
            SnowmanConstants.SMALL_BALL_ON_LARGE_BALL_TOKEN,
            SnowmanConstants.MEDIUM_BALL_ON_LARGE_BALL_TOKEN,
            SnowmanConstants.FULL_SNOW_MAN_TOKEN,
            SnowmanConstants.GRASS_TOKEN,
            SnowmanConstants.SNOW_TOKEN,
            SnowmanConstants.CHARACTER_ON_SNOW_TOKEN,
            SnowmanConstants.CHARACTER_ON_GRASS_TOKEN
        ]
