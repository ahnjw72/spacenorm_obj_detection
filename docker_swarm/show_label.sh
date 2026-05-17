SERVICE=spacenorm_detector
LABEL=server

printf "%-15s %-25s %-47s %s\n" "TASK_ID" "NODE" "TASK_NAME" "LABEL($LABEL)"
printf "%-15s %-25s %-47s %s\n" "---------------" "-------------------------" "---------------------------------------------" "-----------------"

docker service ps "$SERVICE" --format '{{.ID}} {{.Node}} {{.Name}}' | while read id node name; do
  label=$(docker node inspect "$node" --format "{{ index .Spec.Labels \"$LABEL\" }}" 2>/dev/null)
  printf "%-15s %-25s %-47s %s\n" "$id" "$node" "$name" "${label:-<none>}"
done

