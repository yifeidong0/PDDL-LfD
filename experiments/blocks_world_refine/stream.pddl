(define (stream blocks_world)

  (:stream find-grasp
      :inputs (?block)
      :domain (and
          (block ?block) 
      ) 
      :outputs (?X_HB)
      :certified (and
          (handpose ?block ?X_HB)
      )
  )

  (:stream find-table-place
      :inputs (?block ?table) 
      :domain (and
          (block ?block)
          (table ?table) 
      )
      :outputs (?X_WB)
      :certified (and
          (worldpose ?block ?X_WB)    
          (table-support ?block ?X_WB ?table)
      )
  )
  
 (:stream find-block-place
      :inputs (?block ?lowerblock ?X_WL) 
      :domain (and
          (block ?block) 
          (block ?lowerblock) 
          (worldpose ?lowerblock ?X_WL)
      )
      :outputs (?X_WB)
      :certified (and
          (worldpose ?block ?X_WB)
          (block-support ?block ?X_WB ?lowerblock ?X_WL)
      )
  )
)